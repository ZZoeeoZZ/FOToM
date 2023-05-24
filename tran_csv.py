from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm
from tensorboard.backend.event_processing import tag_types

# Legacy aliases
COMPRESSED_HISTOGRAMS = tag_types.COMPRESSED_HISTOGRAMS
HISTOGRAMS = tag_types.HISTOGRAMS
IMAGES = tag_types.IMAGES
AUDIO = tag_types.AUDIO
SCALARS = tag_types.SCALARS
TENSORS = tag_types.TENSORS
GRAPH = tag_types.GRAPH
META_GRAPH = tag_types.META_GRAPH
RUN_METADATA = tag_types.RUN_METADATA

## Normal CDF for std_devs: (-Inf, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, Inf)
## naturally gives bands around median of width 1 std dev, 2 std dev, 3 std dev,
## and then the long tail.
STORE_EVERYTHING_SIZE_GUIDANCE = {
    COMPRESSED_HISTOGRAMS: 0,
    IMAGES: 0,
    AUDIO: 0,
    SCALARS: 0,
    HISTOGRAMS: 0,
    TENSORS: 0,
}


def config():
    parser = argparse.ArgumentParser(description='Export tensorboard data')
    parser.add_argument('--in_path', default='home/zhaozhuoya/maddpg-pytorch/models/simple_adversary/maddpg/run1/logs',
                        type=str, help='Tensorboard event files or a single tensorboard '
                                                                   'file location')
    parser.add_argument('--ex_path', default='home/zhaozhuoya/maddpg-pytorch/results/adv/maddpg/logs',
                        type=str, help='location to save the exported data')

    args = parser.parse_args()

    return args

def main(args):
    # load log data
    # parser = argparse.ArgumentParser(description='Export tensorboard data')
    # parser.add_argument('--in-path', type=str, required=True, help='Tensorboard event files or a single tensorboard '
    #                                                                'file location')
    # parser.add_argument('--ex-path', type=str, required=True, help='location to save the exported data')
    #
    # args = parser.parse_args()

    event_data = event_accumulator.EventAccumulator(args.in_path, size_guidance=STORE_EVERYTHING_SIZE_GUIDANCE)  # a python interface for loading Event data
    event_data.Reload()  # synchronously loads all of the data written so far b
    # print(event_data.Tags())  # print all tags
    keys = event_data.scalars.Keys()  # get all tags,save in a list
    # val = event_data.scalars.Items(keys[1])
    # print(keys)
    df = pd.DataFrame(columns=keys)  # my first column is training loss per iteration, so I abandon it
    for key in tqdm(keys[0:8]):
        # print(key)
        if key != 'train/total_loss_iter':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
            df[key] = pd.DataFrame(event_data.Scalars(key)).value

    df.to_csv(args.ex_path)

    print("Tensorboard data exported successfully")


if __name__ == '__main__':
    args = config()
    # args.in_path = '/home/zhaozhuoya/maddpg-pytorch/models/simple_world_comm/s_o_3/run1/logs/events.out.tfevents.1671939514.biiz-linux'
    # args.in_path = '/home/zhaozhuoya/maddpg-pytorch/models/simple_tag/rs3/run1/logs/events.out.tfevents.1673344036.biiz-linux'
    # args.in_path = '/home/zhaozhuoya/maddpg-pytorch/models/simple_adversary/rs5/run1/logs/events.out.tfevents.1673345155.biiz-linux'
    # args.ex_path = '/home/zhaozhuoya/maddpg-pytorch/results/comm/b3.csv'
    # args.in_path = '/home/zhaozhuoya/maddpg-pytorch/models/simple_tag/rs2/run1/logs/events.out.tfevents.1673255625.biiz-linux'
    args.in_path = '/home/zhaozhuoya/maddpg-pytorch/models/simple_world_comm/rs5/run1/logs/events.out.tfevents.1673344422.biiz-linux'
    args.ex_path = '/home/zhaozhuoya/maddpg-pytorch/results/comm/rs5.csv'
    main(args)

