import argparse  

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Root directory of dataset.")
    parser.add_argument("--model_dir", type=str, required=True, help="The directory of model.")
    parser.add_argument("--batch_size", type=int, default=128, help="Number of batch size.")
    parser.add_argument('--max_length', type=int, default=512, help="the maximum length of the sequence.")
    parser.add_argument('--hidden_size', type=int, default=768, help="The size of hidden layer.")
    parser.add_argument('--num_class', type=int, default=3, help="The number of classes.")

    return parser.parse_args() 