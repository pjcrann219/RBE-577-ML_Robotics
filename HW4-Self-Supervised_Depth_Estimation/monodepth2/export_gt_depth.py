from syndrone_utils import generate_depth_map  # Modify to match your dataset's utilities

def export_gt_depths_syndrone():
    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the Syndrone data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["train", "test"])
    opt = parser.parse_args()

    split_folder = os.path.join(os.path.dirname(__file__), "splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:
        folder, frame_id, _ = line.split()
        frame_id = int(frame_id)

        if opt.split == "train" or opt.split == "test":
            # Modify the path structure based on how Syndrone stores the data
            depth_path = os.path.join(opt.data_path, folder, "depth", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(depth_path)).astype(np.float32) / 256  # Adjust if necessary

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_syndrone()