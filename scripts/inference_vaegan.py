from __future__ import print_function
import open3d as o3d # open3d needs to be imported before other packages!
import argparse
import os
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import pickle

from model.VAE import VAE
from dataset.dataset import RIODatasetSceneGraph, collate_fn_vaegan, collate_fn_vaegan_points
from helpers.util import bool_flag, batch_torch_denormalize_box_params
from helpers.metrics import validate_constrains, validate_constrains_changes, estimate_angular_std
from helpers.visualize_graph import run as vis_graph
from helpers.visualize_scene import render
import helpers.retrieval as retrieval
from model.atlasnet import AE_AtlasNet

# import extension.dist_chamfer as ext
# chamfer = ext.chamferDist()
import json

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=1024, help='number of points in the shape')

parser.add_argument('--dataset', required=False, type=str, default="./GT", help="dataset path")
parser.add_argument('--dataset_3RScan', type=str, default='', help="dataset path of 3RScan")
parser.add_argument('--label_file', required=False, type=str, default='labels.instances.align.annotated.ply', help="label file name")

parser.add_argument('--with_points', type=bool_flag, default=False, help="if false, only predicts layout")
parser.add_argument('--with_feats', type=bool_flag, default=True, help="Load Feats directly instead of points.")

parser.add_argument('--manipulate', default=True, type=bool_flag)
parser.add_argument('--path2atlas', default="./experiments/model_36.pth", type=str)
parser.add_argument('--exp', default='./experiments/layout_test', help='experiment name')
parser.add_argument('--epoch', type=str, default='100', help='saved epoch')
parser.add_argument('--recompute_stats', type=bool_flag, default=False, help='Recomputes statistics of evaluated networks')
parser.add_argument('--evaluate_diversity', type=bool_flag, default=False, help='Computes diversity based on multiple predictions')
parser.add_argument('--visualize', default=False, type=bool_flag)
parser.add_argument('--export_3d', default=False, type=bool_flag, help='Export the generated shapes and boxes in json files for future use')
args = parser.parse_args()

# Function for setting up and running inference
def run_inference():
    print(torch.__version__)

    random.seed(48)
    torch.manual_seed(48)

    argsJson = os.path.join(args.exp, 'args.json')
    assert os.path.exists(argsJson), 'Could not find args.json for experiment {}'.format(args.exp)
    with open(argsJson) as j:
        modelArgs = json.load(j)

    saved_model = torch.load(args.path2atlas)
    point_ae = AE_AtlasNet(num_points=1024, bottleneck_size=128, nb_primitives=25)
    point_ae.load_state_dict(saved_model, strict=True)
    if torch.cuda.is_available():
        point_ae = point_ae.cuda()

    test_dataset_no_changes = RIODatasetSceneGraph(
        root=args.dataset,
        atlas=point_ae,
        path2atlas=args.path2atlas,
        root_3rscan=args.dataset_3RScan,
        label_file=args.label_file,
        split='val_scans',
        npoints=args.num_points,
        data_augmentation=False,
        use_points=args.with_points,
        use_scene_rels=modelArgs['use_scene_rels'],
        vae_baseline=modelArgs['network_type']=='sln',
        with_changes=False,
        eval=True,
        with_feats=args.with_feats,
        use_rio27=modelArgs['rio27'],
        use_canonical=modelArgs['use_canonical'],
        large=modelArgs['large'],
        use_splits=modelArgs['use_splits'],
        crop_floor = False,
        center_scene_to_floor = False)
        # crop_floor=modelArgs['crop_floor'] if 'crop_floor' in modelArgs.keys() else False,
        # center_scene_to_floor=modelArgs['crop_floor'] if 'crop_floor' in modelArgs.keys() else False)

    stats_dataset = RIODatasetSceneGraph(
        root=args.dataset,
        atlas=point_ae,
        path2atlas=args.path2atlas,
        root_3rscan=args.dataset_3RScan,
        label_file=args.label_file,
        npoints=args.num_points,
        split='train_scans',
        use_points=args.with_points,
        use_scene_rels=modelArgs['use_scene_rels'],
        with_changes=False,
        vae_baseline=modelArgs['network_type']=='sln',
        eval=False,
        with_feats=args.with_feats,
        use_rio27=modelArgs['rio27'],
        use_canonical=modelArgs['use_canonical'],
        large=modelArgs['large'],
        use_splits=modelArgs['use_splits'],
        crop_floor = False,
        center_scene_to_floor = False)
        # crop_floor=modelArgs['crop_floor'] if 'crop_floor' in modelArgs.keys() else False,
        # center_scene_to_floor=modelArgs['crop_floor'] if 'crop_floor' in modelArgs.keys() else False)

    if args.with_points:
        collate_fn = collate_fn_vaegan_points
    else:
        collate_fn = collate_fn_vaegan

    test_dataloader_no_changes = torch.utils.data.DataLoader(
        test_dataset_no_changes,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0)

    stats_dataloader = torch.utils.data.DataLoader(
        stats_dataset,
        batch_size=1,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=0)
    model = VAE(type=modelArgs['network_type'], vocab=test_dataset_no_changes.vocab, replace_latent=True,
                with_changes=True, residual=modelArgs['residual'], gconv_pooling=modelArgs['pooling'],
                with_angles=modelArgs['with_angles'])
    model.load_networks(exp=args.exp, epoch=args.epoch)
    
    if torch.cuda.is_available():
        model = model.cuda()
    model = model.eval()
    point_ae = point_ae.eval()

    model.compute_statistics(exp=args.exp, epoch=args.epoch, stats_dataloader=stats_dataloader, force=args.recompute_stats)
    cat2objs = None
    if model.type_ == 'sln':
        # prepare data for retrieval for the 3d-sln baseline
        rel_json_file = args.dataset + '/relationships_merged_train_clean.json'
        cat2objs = retrieval.read_box_json(rel_json_file, args.dataset + '/obj_boxes_train_refined.json')

    def reseed():
        np.random.seed(47)
        torch.manual_seed(47)
        random.seed(47)

    print('\nGeneration Mode')
    validate_constrains_loop(test_dataloader_no_changes, model, with_diversity=args.evaluate_diversity,
                             with_angles=modelArgs['with_angles'], vocab=test_dataset_no_changes.vocab,
                             point_classes_idx=test_dataset_no_changes.point_classes_idx, point_ae=point_ae,
                             export_3d=args.export_3d, cat2objs=cat2objs, datasize='large' if modelArgs['large'] else 'small')


def validate_constrains_loop(testdataloader, model, with_diversity=True, with_angles=False, vocab=None,
                             point_classes_idx=None, point_ae=None, export_3d=False, cat2objs=None, datasize='large',
                             num_samples=10):

    if with_diversity and num_samples < 2:
        raise ValueError('Diversity requires at least two runs (i.e. num_samples > 1).')

    accuracy = {}
    for k in ['left', 'right', 'front', 'behind', 'smaller', 'bigger', 'lower', 'higher', 'same', 'total']:
        # compute validation for these relation categories
        accuracy[k] = []

    all_diversity_boxes = []
    all_diversity_angles = []
    all_diversity_chamfer = []

    all_pred_shapes_exp = {} # for export
    all_pred_boxes_exp = {}

    for i, data in enumerate(testdataloader, 0):
        try:
            dec_objs, dec_triples = data['decoder']['objs'], data['decoder']['tripltes']
            breakpoint()
            instances = data['instance_id'][0]
            scan = data['scan_id'][0]
            split = data['split_id'][0]

        except Exception as e:
            continue

        dec_objs, dec_triples = dec_objs.cuda(), dec_triples.cuda()

        all_pred_boxes = []

        with torch.no_grad():
            boxes_pred, shapes_pred = model.sample_box_and_shape(point_classes_idx, point_ae, dec_objs, dec_triples, attributes=None)
            breakpoint()
            # print(f"!!!!!pred box type{type(boxes_pred)}")
            # print(f"!!!!!pred box{boxes_pred}")
            if with_angles:
                boxes_pred, angles_pred = boxes_pred
                angles_pred = torch.argmax(angles_pred, dim=1, keepdim=True) * 15.0
            else:
                angles_pred = None

            if model.type_ != 'sln':
                shapes_pred, shape_enc_pred = shapes_pred

        if model.type_ != 'sln':
            boxes_pred_den = batch_torch_denormalize_box_params(boxes_pred)
        else:
            boxes_pred_den = boxes_pred

        if export_3d:
            if with_angles:
                boxes_pred_exp = torch.cat([boxes_pred_den.float(),
                                            angles_pred.view(-1,1).float()], 1).detach().cpu().numpy().tolist()
            else:
                boxes_pred_exp = boxes_pred_den.detach().cpu().numpy().tolist()
            if model.type_ != 'sln':
                # save point encodings
                shapes_pred_exp = shape_enc_pred.detach().cpu().numpy().tolist()
            else:
                # 3d-sln baseline does not generate shapes
                # save object labels to use for retrieval instead
                shapes_pred_exp = dec_objs.view(-1,1).detach().cpu().numpy().tolist()
            for i in range(len(shapes_pred_exp)):
                if dec_objs[i] not in testdataloader.dataset.point_classes_idx:
                    shapes_pred_exp[i] = []
            shapes_pred_exp = list(shapes_pred_exp)

            if scan not in all_pred_shapes_exp:
                all_pred_boxes_exp[scan] = {}
                all_pred_shapes_exp[scan] = {}
            if split not in all_pred_shapes_exp[scan]:
                all_pred_boxes_exp[scan][split] = {}
                all_pred_shapes_exp[scan][split] = {}

            all_pred_boxes_exp[scan][split]['objs'] = list(instances)
            all_pred_shapes_exp[scan][split]['objs'] = list(instances)
            for i in range(len(dec_objs) - 1):
                all_pred_boxes_exp[scan][split][instances[i]] = list(boxes_pred_exp[i])
                all_pred_shapes_exp[scan][split][instances[i]] = list(shapes_pred_exp[i])

        if args.visualize:
            # scene graph visualization. saves a picture of each graph to the outfolder
            # print("$$$$$$$$$")
            # print(scan)
            colormap = vis_graph(use_sampled_graphs=False, scan_id=scan, split=str(split), data_path=args.dataset,
                                 outfolder=args.exp + "/vis_graphs/")
            colors = []
            # convert colors to expected format
            def hex_to_rgb(hex):
                hex = hex.lstrip('#')
                hlen = len(hex)
                return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))
            for i in instances:
                h = colormap[str(i)]
                rgb = hex_to_rgb(h)
                colors.append(rgb)
            colors = np.asarray(colors) / 255.

            # layout and shape visualization through open3d
            render_inputs = {
                'boxes_pred_den': boxes_pred_den,
                'angles_pred': angles_pred,
                'classes': vocab['object_idx_to_name'],
                'render_type': 'points',
                'classed_idx': dec_objs,
                'shapes_pred': shapes_pred.cpu().detach(),
                'colors': colors,
                'render_boxes': True,
                'output_path': args.exp + "/vis_scenes/"
                }
            with open('render_inputs.pkl', 'wb') as f:
                pickle.dump(render_inputs, f)
                print("render inputs saved!!!!")
            render(boxes_pred_den, angles_pred, classes=vocab['object_idx_to_name'], render_type='points', classed_idx=dec_objs,
                   shapes_pred=shapes_pred.cpu().detach(), colors=colors, render_boxes=True)

        all_pred_boxes.append(boxes_pred_den.cpu().detach())
        if with_diversity:

            # Run multiple times to obtain diversities
            # Diversity results for this dataset sample
            boxes_diversity_sample, shapes_sample, angle_diversity_sample, diversity_retrieval_ids_sample = [], [], [], []
            for sample in range(num_samples):
                diversity_boxes, diversity_points = model.sample_box_and_shape(point_classes_idx, point_ae, dec_objs, dec_triples,
                                                                               attributes=None)
                if with_angles:
                    diversity_boxes, diversity_angles = diversity_boxes
                if model.type_ == 'sln':
                    diversity_points, diversity_retrieval_ids = retrieval.rio_retrieve(
                        dec_objs, diversity_boxes, vocab, cat2objs, testdataloader.dataset.root_3rscan,
                        return_retrieval_id=True)
                else:
                    diversity_points = diversity_points[0]

                # Computing shape diversity on canonical and normalized shapes
                normalized_points = []
                filtered_diversity_retrieval_ids = []
                for ins_id, obj_id in enumerate(dec_objs):
                    if obj_id != 0 and obj_id in testdataloader.dataset.point_classes_idx:
                        points = diversity_points[ins_id]
                        if type(points) is torch.Tensor:
                            points = points.cpu().numpy()
                        if points is None:
                            continue
                        # Normalizing shapes
                        points = torch.from_numpy(normalize(points))
                        if torch.cuda.is_available():
                            points = points.cuda()
                        normalized_points.append(points)
                        if model.type_ == 'sln':
                            filtered_diversity_retrieval_ids.append(diversity_retrieval_ids[ins_id])

                # We use keep to filter changed nodes
                boxes_diversity_sample.append(diversity_boxes)

                if with_angles:
                    # We use keep to filter changed nodes
                    angle_diversity_sample.append(np.expand_dims(np.argmax(diversity_angles.cpu().numpy(), 1), 1) / 24. * 360.)

                if len(normalized_points) > 0:
                    shapes_sample.append(torch.stack(normalized_points)) # keep has already been aplied for points
                    if model.type_ == 'sln':
                        diversity_retrieval_ids_sample.append(np.stack(filtered_diversity_retrieval_ids))


            # # Compute standard deviation for box for this sample
            # if len(boxes_diversity_sample) > 0:
            #     boxes_diversity_sample = torch.stack(boxes_diversity_sample, 1)
            #     bs = boxes_diversity_sample.shape[0]
            #     if model.type_ != 'sln':
            #         boxes_diversity_sample = batch_torch_denormalize_box_params(boxes_diversity_sample.reshape([-1, 6])).reshape([bs, -1, 6])
            #     all_diversity_boxes += torch.std(boxes_diversity_sample, dim=1).cpu().numpy().tolist()

            #     # Compute standard deviation for angle for this sample
            # if len(angle_diversity_sample) > 0:
            #     angle_diversity_sample = np.stack(angle_diversity_sample, 1)
            #     all_diversity_angles += [estimate_angular_std(d[:,0]) for d in angle_diversity_sample]

            #     # Compute chamfer distances for shapes for this sample
            # if len(shapes_sample) > 0:
            #     shapes_sample = torch.stack(shapes_sample, 1)
            #     for shapes_id in range(len(shapes_sample)):
            #         # Taking a single predicted shape
            #         shapes = shapes_sample[shapes_id]
            #         if len(diversity_retrieval_ids_sample) > 0:
            #             # To avoid that retrieval the object ids like 0,1,0,1,0 gives high error
            #             # We sort them to measure how often different objects are retrieved 0,0,0,1,1
            #             diversity_retrieval_ids = diversity_retrieval_ids_sample[shapes_id]
            #             sorted_idx = diversity_retrieval_ids.argsort()
            #             shapes = shapes[sorted_idx]
            #         sequence_diversity = []
            #         # Iterating through its multiple runs
            #         for shape_sequence_id in range(len(shapes) - 1):
            #             # Compute chamfer with the next shape in its sequences
            #             dist1, dist2 = chamfer(shapes[shape_sequence_id:shape_sequence_id + 1].float(),
            #                                    shapes[shape_sequence_id + 1:shape_sequence_id + 2].float())
            #             chamfer_dist = torch.mean(dist1) + torch.mean(dist2)
            #             # Save the distance
            #             sequence_diversity += [chamfer_dist.cpu().numpy().tolist()]

            #         if len(sequence_diversity) > 0:  # check if sequence has shapes
            #             all_diversity_chamfer.append(np.mean(sequence_diversity))

        # compute constraints accuracy through simple geometric rules
        # accuracy = validate_constrains(dec_triples, boxes_pred, None, None, model.vocab, accuracy, with_norm=model.type_ != 'sln')

    if export_3d:
        # export box and shape predictions for future evaluation
        result_path = os.path.join(args.exp, 'results')
        if not os.path.exists(result_path):
            # Create a new directory for results
            os.makedirs(result_path)
        shape_filename = os.path.join(result_path, 'shapes_' + ('large' if datasize else 'small') + '.json')
        box_filename = os.path.join(result_path, 'boxes_' + ('large' if datasize else 'small') + '.json')
        json.dump(all_pred_boxes_exp, open(box_filename, 'w')) # 'dis_nomani_boxes_large.json'
        json.dump(all_pred_shapes_exp, open(shape_filename, 'w'))

    # if with_diversity:
    #     print("DIVERSITY:")
    #     print("\tShape (Avg. Chamfer Distance) = %f" % (np.mean(all_diversity_chamfer)))
    #     print("\tBox (Std. metric size and location) = %f, %f" % (
    #         np.mean(np.mean(all_diversity_boxes, axis=0)[:3]),
    #         np.mean(np.mean(all_diversity_boxes, axis=0)[3:])))
    #     print("\tAngle (Std.) %s = %f" % (k, np.mean(all_diversity_angles)))

    # keys = list(accuracy.keys())
    # for dic, typ in [(accuracy, "acc")]:

    #     print('{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} &{:.2f} & {:.2f}'.format(typ, np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])]),
    #                                                                            np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])]), np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])]),
    #                                                                            np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])]), np.mean(dic[keys[8]]), np.mean(dic[keys[9]])))
    #     print('means of mean: {:.2f}'.format(np.mean([np.mean([np.mean(dic[keys[0]]), np.mean(dic[keys[1]])]),
    #                                                   np.mean([np.mean(dic[keys[2]]), np.mean(dic[keys[3]])]), np.mean([np.mean(dic[keys[4]]), np.mean(dic[keys[5]])]),
    #                                                   np.mean([np.mean(dic[keys[6]]), np.mean(dic[keys[7]])]), np.mean(dic[keys[8]])])))


def normalize(vertices, scale=1):
    xmin, xmax = np.amin(vertices[:, 0]), np.amax(vertices[:, 0])
    ymin, ymax = np.amin(vertices[:, 1]), np.amax(vertices[:, 1])
    zmin, zmax = np.amin(vertices[:, 2]), np.amax(vertices[:, 2])

    vertices[:, 0] += -xmin - (xmax - xmin) * 0.5
    vertices[:, 1] += -ymin - (ymax - ymin) * 0.5
    vertices[:, 2] += -zmin - (zmax - zmin) * 0.5

    scalars = np.max(vertices, axis=0)
    scale = scale

    vertices = vertices / scalars * scale
    return vertices


# Example usage
# dataset = <initialize your dataset here>
# run_inference(model_path='./experiments/model_36.pth', dataset=dataset)

if __name__ == "__main__": run_inference()