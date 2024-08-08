import os
import cv2
import numpy as np
import struct
import torch
import lib.ransac_voting.ransac_voting as ransac_voting
# from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer_v3
import matplotlib.pyplot as plt


class PostProcess:
    def __init__(self, results, ** kwargs):
        self.model = self.load_ply(results['model_path'])['pts'] / 1000
        self.corners_3d = self.get_model_corners()
        self.K = results['K']
        self.kpts_3d = results['kpts']
        self.results = results

    def processResults(self, results):
        self.time = results['time']
        self.results['img_path'] = results['img_path']
        mask = torch.from_numpy(results['output'][2]).cuda()
        self.mask = mask
        vertex = torch.from_numpy(results['output'][1]).permute(0, 2, 3, 1)
        b, h, w, vn_2 = vertex.shape
        vertex = vertex.view(b, h, w, vn_2 // 2, 2).cuda()
        self.vertex = vertex
        kpts = self.ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
        # kpts = ransac_voting_layer_v3(mask, vertex, 128, inlier_thresh=0.99, max_num=100)
        self.kpts_2d_pred = kpts.cpu().numpy()[0][:8, :]

    def project(self, xyz, RT):
        """
        xyz: [N, 3]
        K: [3, 3]
        RT: [3, 4]
        """
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, self.K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy

    def draw(self, single=True):
        img = cv2.imread(self.results['img_path'][0])
        # self.visualize_hypothesis(np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        pose = self.pnp()
        self.corners_2d = self.project(self.corners_3d, pose)
        xy1 = self.corners_2d[[0, 1, 3, 2, 0, 4, 6, 2]]
        xy2 = self.corners_2d[[5, 4, 6, 7, 5, 1, 3, 7]]
        cv2.polylines(img, np.int32([xy1]), True, (0, 0, 255), 2)
        cv2.polylines(img, np.int32([xy2]), True, (0, 0, 255), 2)
        cv2.putText(img, '{:.2f}ms'.format(self.time), (40, 40), 0,
                    fontScale=1, color=(0, 0, 0), thickness=2)
        if single:
            cv2.imshow('Image', img)
            if cv2.waitKey(0) == ord('q'):
                cv2.destroyAllWindows()
        else:
            cv2.imshow('Image', img)
            cv2.imshow('mask', self.results['output'][2][0].astype(np.uint8) * 255)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()

    def pnp(self, method=cv2.SOLVEPNP_EPNP):
        points_3d = self.kpts_3d.copy()
        points_2d = self.kpts_2d_pred.copy()
        camera_matrix = self.K.copy()
        try:
            dist_coeffs = self.pnp.dist_coeffs
        except BaseException:
            dist_coeffs = np.zeros(shape=[8, 1], dtype='float64')
        assert points_3d.shape[0] == points_2d.shape[0], 'points 3D and points 2D must have same number of vertices'
        if method == cv2.SOLVEPNP_EPNP:
            points_3d = np.expand_dims(points_3d, 0)
            points_2d = np.expand_dims(points_2d, 0)

        points_2d = np.ascontiguousarray(points_2d .astype(np.float64))
        points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
        camera_matrix = camera_matrix.astype(np.float64)

        _, R_exp, t = cv2.solvePnP(points_3d,
                                   points_2d,
                                   camera_matrix,
                                   dist_coeffs,
                                   flags=method)
        R, _ = cv2.Rodrigues(R_exp)
        return np.concatenate([R, t], axis=-1)

    def ransac_voting_layer_v3(self, mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                               min_num=5, max_num=30000):
        '''
        :param mask:      [b,h,w]
        :param vertex:    [b,h,w,vn,2]
        :param round_hyp_num:
        :param inlier_thresh:
        :return: [b,vn,2]
        '''
        b, h, w, vn, _ = vertex.shape
        batch_win_pts = []
        for bi in range(b):
            hyp_num = 0
            cur_mask = (mask[bi]).byte()
            foreground_num = torch.sum(cur_mask)

            # if too few points, just skip it
            if foreground_num < min_num:
                win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
                batch_win_pts.append(win_pts)  # [1,vn,2]
                continue

            # if too many inliers, we randomly down sample it
            if foreground_num > max_num:
                selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
                selected_mask = (selection < (max_num / foreground_num.float())).byte()
                cur_mask *= selected_mask

            coords = torch.nonzero(cur_mask).float()  # [tn,2]
            coords = coords[:, [1, 0]]
            direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
            direct = direct.view([coords.shape[0], vn, 2])
            tn = coords.shape[0]
            idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
            all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
            all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

            cur_iter = 0
            while True:
                # generate hypothesis
                cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

                # voting for hypothesis
                cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
                ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

                # find max
                cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]
                cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
                cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
                cur_win_ratio = cur_win_counts.float() / tn

                # update best point
                larger_mask = all_win_ratio < cur_win_ratio
                all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
                all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]

                # check confidence
                hyp_num += round_hyp_num
                cur_iter += 1
                cur_min_ratio = torch.min(all_win_ratio)
                if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                    break

            # compute mean intersection again
            normal = torch.zeros_like(direct)   # [tn,vn,2]
            normal[:, :, 0] = direct[:, :, 1]
            normal[:, :, 1] = -direct[:, :, 0]
            all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=mask.device)
            all_win_pts = torch.unsqueeze(all_win_pts, 0)  # [1,vn,2]
            ransac_voting.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)  # [1,vn,tn]

            # coords [tn,2] normal [vn,tn,2]
            all_inlier = torch.squeeze(all_inlier.float(), 0)              # [vn,tn]
            normal = normal.permute(1, 0, 2)                                # [vn,tn,2]
            normal = normal * torch.unsqueeze(all_inlier, 2)                 # [vn,tn,2] outlier is all zero

            b = torch.sum(normal * torch.unsqueeze(coords, 0), 2)             # [vn,tn]
            ATA = torch.matmul(normal.permute(0, 2, 1), normal)              # [vn,2,2]
            ATb = torch.sum(normal * torch.unsqueeze(b, 2), 1)                # [vn,2]
            # try:
            all_win_pts = torch.matmul(self.b_inv(ATA), torch.unsqueeze(ATb, 2))  # [vn,2,1]
            # except:
            #    __import__('ipdb').set_trace()
            batch_win_pts.append(all_win_pts[None, :, :, 0])

        batch_win_pts = torch.cat(batch_win_pts)
        return batch_win_pts

    def produce_hypothesis(self, mask, vertex, round_hyp_num, inlier_thresh=0.999, confidence=0.99, max_iter=20,
                           min_num=5, max_num=30000):
        '''
        :param mask:      [b,h,w]
        :param vertex:    [b,h,w,vn,2]
        :param round_hyp_num:
        :param inlier_thresh:
        :return: [b,vn,2]
        '''
        b, h, w, vn, _ = vertex.shape
        batch_hyp_pts = []
        batch_hyp_counts = []
        for bi in range(b):
            hyp_num = 0
            cur_mask = (mask[bi]).byte()
            foreground_num = torch.sum(cur_mask)

            # if too few points, just skip it
            if foreground_num < min_num:
                win_pts = torch.zeros([1, vn, 2], dtype=torch.float32, device=mask.device)
                batch_win_pts.append(win_pts)  # [1,vn,2]
                continue

            # if too many inliers, we randomly down sample it
            if foreground_num > max_num:
                selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=mask.device).uniform_(0, 1)
                selected_mask = (selection < (max_num / foreground_num.float()))
                cur_mask *= selected_mask

            coords = torch.nonzero(cur_mask).float()  # [tn,2]
            coords = coords[:, [1, 0]]
            direct = vertex[bi].masked_select(torch.unsqueeze(torch.unsqueeze(cur_mask, 2), 3))  # [tn,vn,2]
            direct = direct.view([coords.shape[0], vn, 2])
            tn = coords.shape[0]
            idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=mask.device).random_(0, direct.shape[0])
            all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=mask.device)
            all_win_pts = torch.zeros([vn, 2], dtype=torch.float32, device=mask.device)

            # generate hypothesis
            cur_hyp_pts = ransac_voting.generate_hypothesis(direct, coords, idxs)  # [hn,vn,2]

            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=mask.device)
            ransac_voting.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)  # [hn,vn,tn]

            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn,vn]

            batch_hyp_pts.append(cur_hyp_pts)
            batch_hyp_counts.append(cur_inlier_counts)

        return torch.stack(batch_hyp_pts), torch.stack(batch_hyp_counts)

    def get_model_corners(self):
        min_x, max_x = np.min(self.model[:, 0]), np.max(self.model[:, 0])
        min_y, max_y = np.min(self.model[:, 1]), np.max(self.model[:, 1])
        min_z, max_z = np.min(self.model[:, 2]), np.max(self.model[:, 2])
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])
        return corners_3d

    def load_ply(self, ply_path):
        """ Loads a 3D mesh model from a PLY file.
        :return: The loaded model given by a dictionary with items:
        'pts' (nx3 ndarray), 'normals' (nx3 ndarray), 'colors' (nx3 ndarray),
        'faces' (mx3 ndarray) - the latter three are optional.
        """
        f = open(ply_path, 'r')

        n_pts = 0
        n_faces = 0
        face_n_corners = 3  # Only triangular faces are supported
        pt_props = []
        face_props = []
        is_binary = False
        header_vertex_section = False
        header_face_section = False

        # Read header
        while True:
            line = f.readline().rstrip('\n').rstrip('\r')  # Strip the newline character(s)
            if line.startswith('element vertex'):
                n_pts = int(line.split()[-1])
                header_vertex_section = True
                header_face_section = False
            elif line.startswith('element face'):
                n_faces = int(line.split()[-1])
                header_vertex_section = False
                header_face_section = True
            elif line.startswith('element'):  # Some other element
                header_vertex_section = False
                header_face_section = False
            elif line.startswith('property') and header_vertex_section:
                # (name of the property, data type)
                pt_props.append((line.split()[-1], line.split()[-2]))
            elif line.startswith('property list') and header_face_section:
                elems = line.split()
                if elems[-1] == 'vertex_indices':
                    # (name of the property, data type)
                    face_props.append(('n_corners', elems[2]))
                    for i in range(face_n_corners):
                        face_props.append(('ind_' + str(i), elems[3]))
                else:
                    print('Warning: Not supported face property: ' + elems[-1])
            elif line.startswith('format'):
                if 'binary' in line:
                    is_binary = True
            elif line.startswith('end_header'):
                break

        # Prepare data structures
        model = {}
        model['pts'] = np.zeros((n_pts, 3), float)
        if n_faces > 0:
            model['faces'] = np.zeros((n_faces, face_n_corners), float)

        pt_props_names = [p[0] for p in pt_props]
        is_normal = False
        if {'nx', 'ny', 'nz'}.issubset(set(pt_props_names)):
            is_normal = True
            model['normals'] = np.zeros((n_pts, 3), float)

        is_color = False
        model['colors'] = np.zeros((n_pts, 3), float)
        if {'red', 'green', 'blue'}.issubset(set(pt_props_names)):
            is_color = True
            model['colors'] = np.zeros((n_pts, 3), float)

        is_texture = False
        if {'texture_u', 'texture_v'}.issubset(set(pt_props_names)):
            is_texture = True
            model['texture_uv'] = np.zeros((n_pts, 2), float)

        formats = {  # For binary format
            'float': ('f', 4),
            'double': ('d', 8),
            'int': ('i', 4),
            'uchar': ('B', 1)
        }

        # Load vertices
        for pt_id in range(n_pts):
            prop_vals = {}
            load_props = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                          'red', 'green', 'blue', 'texture_u', 'texture_v']
            if is_binary:
                for prop in pt_props:
                    format = formats[prop[1]]
                    val = struct.unpack(format[0], f.read(format[1]))[0]
                    if prop[0] in load_props:
                        prop_vals[prop[0]] = val
            else:
                elems = f.readline().rstrip('\n').rstrip('\r').split()
                for prop_id, prop in enumerate(pt_props):
                    if prop[0] in load_props:
                        prop_vals[prop[0]] = elems[prop_id]

            model['pts'][pt_id, 0] = float(prop_vals['x'])
            model['pts'][pt_id, 1] = float(prop_vals['y'])
            model['pts'][pt_id, 2] = float(prop_vals['z'])

            if is_normal:
                model['normals'][pt_id, 0] = float(prop_vals['nx'])
                model['normals'][pt_id, 1] = float(prop_vals['ny'])
                model['normals'][pt_id, 2] = float(prop_vals['nz'])

            if is_color:
                model['colors'][pt_id, 0] = float(prop_vals['red'])
                model['colors'][pt_id, 1] = float(prop_vals['green'])
                model['colors'][pt_id, 2] = float(prop_vals['blue'])

            if is_texture:
                model['texture_uv'][pt_id, 0] = float(prop_vals['texture_u'])
                model['texture_uv'][pt_id, 1] = float(prop_vals['texture_v'])

        # Load faces
        for face_id in range(n_faces):
            prop_vals = {}
            if is_binary:
                for prop in face_props:
                    format = formats[prop[1]]
                    val = struct.unpack(format[0], f.read(format[1]))[0]
                    if prop[0] == 'n_corners':
                        if val != face_n_corners:
                            print('Error: Only triangular faces are supported.')
                            print('Number of face corners: ' + str(val))
                            exit(-1)
                    else:
                        prop_vals[prop[0]] = val
            else:
                elems = f.readline().rstrip('\n').rstrip('\r').split()
                for prop_id, prop in enumerate(face_props):
                    if prop[0] == 'n_corners':
                        if int(elems[prop_id]) != face_n_corners:
                            print('Error: Only triangular faces are supported.')
                            print('Number of face corners: ' + str(int(elems[prop_id])))
                            exit(-1)
                    else:
                        prop_vals[prop[0]] = elems[prop_id]

            model['faces'][face_id, 0] = int(prop_vals['ind_0'])
            model['faces'][face_id, 1] = int(prop_vals['ind_1'])
            model['faces'][face_id, 2] = int(prop_vals['ind_2'])

        f.close()
        model['pts'] *= 1000.

        return model

    def draw_hypothesis(self, rgb, hyp_pts, hyp_counts, pts_target=None, save=False, save_fn=None):
        '''
        :param rgb:         b,h,w
        :param hyp_pts:     b,hn,vn,2
        :param hyp_counts:  b,hn,vn
        :param pts_target:  b,vn,2
        :param save:
        :param save_fn:
        :return:
        '''
        b, hn, vn, _ = hyp_pts.shape
        h, w, _ = rgb.shape
        for bi in range(b):
            for vi in range(vn):
                cur_hyp_counts = hyp_counts[bi, :, vi]  # [hn]
                cur_hyp_pts = hyp_pts[bi, :, vi]        # [hn,2]
                # mask=np.logical_and(np.logical_and(cur_hyp_pts[:,0]>-w*0.5,cur_hyp_pts[:,0]<w*1.5),
                #                     np.logical_and(cur_hyp_pts[:,1]>-h*0.5,cur_hyp_pts[:,1]<h*1.5))
                mask = np.logical_and(np.logical_and(cur_hyp_pts[:, 0] > 0, cur_hyp_pts[:, 0] < w * 1.0),
                                      np.logical_and(cur_hyp_pts[:, 1] > 0, cur_hyp_pts[:, 1] < h * 1.0))
                cur_hyp_pts[np.logical_not(mask)] = 0.0
                cur_hyp_counts[np.logical_not(mask)] = 0

                cur_hyp_counts = cur_hyp_counts.astype(np.float32)
                colors = (cur_hyp_counts / cur_hyp_counts.max())  # [:,None]#*np.array([[255,0,0]])
                plt.figure(figsize=(10, 8))
                plt.imshow(rgb)
                plt.scatter(cur_hyp_pts[:, 0], cur_hyp_pts[:, 1], c=colors, s=0.1, cmap='viridis')
                # plt.plot(pts_target[bi,vi,0],pts_target[bi,vi,1],'*',c='r')
                if save:
                    plt.savefig(save_fn.format(bi, vi))
                else:
                    plt.show()
                plt.close()

    def visualize_hypothesis(self, image):
        idxs = torch.zeros([128, 9, 2], dtype=torch.int32, device=0).random_(0, 94)
        mask_contiguous = self.mask.contiguous()
        vertex_contiguous = self.vertex.contiguous()
        idxs_contiguous = idxs.contiguous()
        hyp, hyp_counts = self.produce_hypothesis(mask_contiguous, vertex_contiguous, 128)
        # image = self.imagenet_to_uint8(image)
        hyp = hyp.detach().cpu().numpy()
        hyp_counts = hyp_counts.detach().cpu().numpy()
        self.draw_hypothesis(image, hyp, hyp_counts)

    def imagenet_to_uint8(self, rgb, torch_format=True):
        '''
        :param rgb: [b,3,h,w]
        :return:
        '''
        if torch_format:
            if len(rgb.shape) == 4:
                rgb = rgb.transpose(0, 2, 3, 1)
            else:
                rgb = rgb.transpose(1, 2, 0)
        rgb *= np.asarray([0.229, 0.224, 0.225])[None, None, :]
        rgb += np.asarray([0.485, 0.456, 0.406])[None, None, :]
        rgb *= 255
        rgb = rgb.astype(np.uint8)

        return rgb

    def b_inv(self, b_mat):
        '''
        code from
        https://stackoverflow.com/questions/46595157/how-to-apply-the-torch-inverse-function-of-pytorch-to-every-sample-in-the-batc
        :param b_mat:
        :return:
        '''
        eye = b_mat.new_ones(b_mat.size(-1)).diag().expand_as(b_mat)
        try:
            # b_inv, _ = torch.solve(eye, b_mat)
            b_inv = torch.linalg.solve(b_mat, eye)
        except BaseException:
            b_inv = eye
        return b_inv
