import argparse
import os


def argparser():
  parser = argparse.ArgumentParser()
  # for model
  parser.add_argument(
      '--seq_window_lengths',
      type=int,
      nargs='+',
      help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)'
  )
  parser.add_argument(
      '--smi_window_lengths',
      type=int,
      nargs='+',
      help='Space seperated list of motif filter lengths. (ex, --window_lengths 4 8 12)'
  )
  parser.add_argument(
      '--num_windows',
      type=int,
      nargs='+',
      help='Space seperated list of the number of motif filters corresponding to length list. (ex, --num_windows 100 200 100)'
  )

  parser.add_argument(
      '--max_seq_len',
      type=int,
      default=1000,
      help='Length of input sequences.'
  )
  parser.add_argument(
      '--max_smi_len',
      type=int,
      default=100,
      help='Length of input sequences.'
  )

  parser.add_argument(
      '--num_epoch',
      type=int,
      default=100,
      help='Number of epochs to train.'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=256,
      help='Batch size. Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--dataset_path',
      type=str,
      default='./data/kiba/',
      help='Directory for input data.'
  )
  parser.add_argument(
      '--problem_type',
      type=int,
      default=2,
      help='Type of the prediction problem (1-4)'
  )

  parser.add_argument(
      '--is_log',
      type=int,
      default=0,
      help='use log transformation for Y'
  )
  parser.add_argument(
      '--checkpoint_path',
      type=str,
      default='',
      help='Path to write checkpoint file.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default='/tmp',
      help='Directory for log data.'
  )
  parser.add_argument(
      '--lamda',
      type=int,
      default=-5,
      nargs='+',
  )


  FLAGS, unparsed = parser.parse_known_args()

  return FLAGS




def logging(msg, FLAGS):
  fpath = os.path.join( FLAGS.log_dir, "log.txt" )
  with open( fpath, "a" ) as fw:
    fw.write("%s\n" % msg)
  #print(msg)
