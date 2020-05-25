# -*- coding: UTF-8 -*-
"""TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import sys

import tensorflow as tf  # pylint: disable=g-bad-import-order

from official.utils.arg_parsers import parsers
from official.utils.logs import hooks_helper
from official.utils.misc import model_helpers

_DIR_CONFIG = {
    'data_dir' :'/home/ubuntu/test/data',
    'model_dir' : '/home/ubuntu/test/model',
    #'train_file':'adult.data',
    #'test_file':'adult.test' #'adult.test'
    'train_file':'trainval_wkfpd30_subset5.csv',
    'test_file':'adult.data'
}

_NUM_EXAMPLES = {
    #'train': 20000,
    #'validation': 3947,
    'train': 165236,
    'validation':20000
}

_CSV_COLUMNS = [
    "customerid", "data_partition", "fpd30", "personal_info_certid",
    "personal_info_sex", "personal_info_age", "personal_info_industry",
    "personal_info_province", "personal_info_edu", "personal_info_marriage",
    "credit_card_num", "credit_bank_num", "credit_limit_avg",
    "credit_limit_total", "credit_limit_max", "credit_limit_min",
    "operator_carrier", "operator_time_to_auth", "operator_time_to_2000",
    "operator_duration_one_month", "operator_dial_num_one_month",
    "operator_dial_num_three_month", "operator_num_one_month", "operator_num_three_month",
    "operator_duration_avg", "operator_duration_stddev", "operator_dial_duration_percent",
    "operator_dial_num_percent", "operator_dial_0_6_dur_per",
    "operator_dial_0_6_num_per", "operator_dial_6_12_dur_per", "operator_dial_6_12_num_per",
    "operator_dial_12_18_dur_per", "operator_dial_12_18_num_per", "operator_dial_18_24_dur_per",
    "operator_dial_18_24_num_per", "operator_dial_work_dur_per", "operator_dial_work_num_per",
    "devinfo_devicetype", "devinfo_brand", "devinfo_model", "debit_regionalmobility",
    "debit_capacity", "debit_transmonthcount", "debit_residentcity", "debit_transmostarea",
    "debit_avgamountmonthly", "debit_avgcountmonthly", "debit_diffday", "debit_maxtransinterval"
]

_CSV_COLUMN_DEFAULTS = [[''], [''], [''], [''], [''], [0], [''], [''], [''], [''],
                         [0], [0], [0.0], [0.0], [0.0], [0.0], [''], [0], [0], [0.0],
                         [0], [0], [0], [0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0],
                         [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [''], [''],
                         [''], [0], [0.0], [0], [''], [''], [0.0], [0.0], [0], [0]
]

LOSS_PREFIX = {'wide': 'linear/', 'deep': 'dnn/'}


# ## wide&deep model
def build_model_columns():
    """Builds a set of wide and deep feature columns."""
    print('............................')
    #  Continuous columns
    personal_info_sex = tf.feature_column.categorical_column_with_vocabulary_list(
    'personal_info_sex', ['0', '1'])
    personal_info_age = tf.feature_column.numeric_column('personal_info_age')
    # Transformation personal_info_age
    age_buckets = tf.feature_column.bucketized_column(
      personal_info_age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # categorical from file
    personal_info_industry = tf.feature_column.categorical_column_with_vocabulary_file(
    'personal_info_industry', 'personal_info_industry.txt')
    personal_info_province = tf.feature_column.categorical_column_with_vocabulary_file(
    'personal_info_province', 'province.txt')
    personal_info_edu = tf.feature_column.categorical_column_with_vocabulary_list(
    'personal_info_edu', [
      'B0301', 'B0302', 'B0303', 'B0304', 'B0305', 'B0306'
    ])
    personal_info_marriage = tf.feature_column.categorical_column_with_vocabulary_list(
    'personal_info_marriage', ['B0501', 'B0502', 'B0503', 'B0504', 'B0505', 'B0506'])
    credit_card_num = tf.feature_column.numeric_column('credit_card_num')
    bucketized_credit_card_num = tf.feature_column.bucketized_column(
      credit_card_num, boundaries=[1, 3, 6])
    credit_bank_num = tf.feature_column.numeric_column('credit_bank_num')
    credit_limit_avg = tf.feature_column.numeric_column('credit_limit_avg')
    credit_limit_total = tf.feature_column.numeric_column('credit_limit_total')
    credit_limit_max = tf.feature_column.numeric_column('credit_limit_max')
    credit_limit_min = tf.feature_column.numeric_column('credit_limit_min')
    operator_carrier = tf.feature_column.categorical_column_with_vocabulary_list(
    'operator_carrier', [
      'CHINA_UNICOM', 'CHINA_MOBILE', '联通', 'CHINA_TELECOM', '电信', '移动'
    ])

    operator_time_to_auth = tf.feature_column.numeric_column('operator_time_to_auth')
    operator_time_to_2000 = tf.feature_column.numeric_column('operator_time_to_2000')
    operator_duration_one_month = tf.feature_column.numeric_column('operator_duration_one_month')
    operator_dial_num_one_month = tf.feature_column.numeric_column('operator_dial_num_one_month')
    operator_dial_num_three_month = tf.feature_column.numeric_column('operator_dial_num_three_month')
    operator_num_one_month = tf.feature_column.numeric_column('operator_num_one_month')
    operator_num_three_month = tf.feature_column.numeric_column('operator_num_three_month')
    operator_duration_avg = tf.feature_column.numeric_column('operator_duration_avg')
    operator_duration_stddev = tf.feature_column.numeric_column('operator_duration_stddev')
    operator_dial_duration_percent = tf.feature_column.numeric_column('operator_dial_duration_percent')
    operator_dial_num_percent = tf.feature_column.numeric_column('operator_dial_num_percent')
    operator_dial_0_6_dur_per = tf.feature_column.numeric_column('operator_dial_0_6_dur_per')
    operator_dial_0_6_num_per = tf.feature_column.numeric_column('operator_dial_0_6_num_per')
    bucketized_operator_dial_0_6_num_per = tf.feature_column.bucketized_column(
      operator_dial_0_6_num_per, boundaries=[1, 3, 6])
    operator_dial_6_12_dur_per = tf.feature_column.numeric_column('operator_dial_6_12_dur_per')
    operator_dial_6_12_num_per = tf.feature_column.numeric_column('operator_dial_6_12_num_per')
    bucketized_operator_dial_6_12_num_per = tf.feature_column.bucketized_column(
      operator_dial_6_12_num_per, boundaries=[1, 3, 6])
    operator_dial_12_18_dur_per = tf.feature_column.numeric_column('operator_dial_12_18_dur_per')
    operator_dial_12_18_num_per = tf.feature_column.numeric_column('operator_dial_12_18_num_per')
    bucketized_operator_dial_12_18_num_per = tf.feature_column.bucketized_column(
      operator_dial_12_18_num_per, boundaries=[1, 3, 6])
    operator_dial_18_24_dur_per = tf.feature_column.numeric_column('operator_dial_18_24_dur_per')
    operator_dial_18_24_num_per = tf.feature_column.numeric_column('operator_dial_18_24_num_per')
    bucketized_operator_dial_18_24_num_per = tf.feature_column.bucketized_column(
      operator_dial_18_24_num_per, boundaries=[1, 3, 6])
    operator_dial_work_dur_per = tf.feature_column.numeric_column('operator_dial_work_dur_per')
    operator_dial_work_num_per = tf.feature_column.numeric_column('operator_dial_work_num_per')
    devinfo_devicetype = tf.feature_column.categorical_column_with_vocabulary_list(
    'devinfo_devicetype', [
        'android', 'ios', 'android_h5', 'ios_h5', 'h5'])
    devinfo_brand = tf.feature_column.categorical_column_with_vocabulary_file(
    'devinfo_brand', 'devinfo_brand.txt')
    devinfo_model = tf.feature_column.categorical_column_with_vocabulary_file(
    'devinfo_model', 'devinfo_model.txt')
    debit_regionalmobility = tf.feature_column.numeric_column('debit_regionalmobility')
    debit_capacity = tf.feature_column.numeric_column('debit_capacity')
    debit_transmonthcount = tf.feature_column.numeric_column('debit_transmonthcount')
    debit_residentcity = tf.feature_column.categorical_column_with_vocabulary_file(
    'debit_residentcity', 'debit_residentcity.txt')
    debit_transmostarea = tf.feature_column.categorical_column_with_vocabulary_list(
    'debit_transmostarea', [
        '一线城市', '二线城市', '三线城市', '其他城市', '境外'])
    debit_avgamountmonthly = tf.feature_column.numeric_column('debit_avgamountmonthly')
    debit_avgcountmonthly = tf.feature_column.numeric_column('debit_avgcountmonthly')
    debit_diffday = tf.feature_column.numeric_column('debit_diffday')
    debit_maxtransinterval = tf.feature_column.numeric_column('debit_maxtransinterval')

    # Wide columns and deep columns.
    base_columns = [
        personal_info_sex,
        age_buckets,
        personal_info_industry,
        personal_info_province,
        personal_info_edu,
        personal_info_marriage,
        operator_carrier,
        devinfo_devicetype,
        devinfo_brand,
        devinfo_model,
        debit_residentcity,
        debit_transmostarea,
	bucketized_operator_dial_0_6_num_per,
	bucketized_operator_dial_6_12_num_per,
	bucketized_operator_dial_12_18_num_per,
	bucketized_operator_dial_18_24_num_per,
    ]

    crossed_columns = [
	tf.feature_column.crossed_column(['personal_info_sex', age_buckets, 'personal_info_industry', 'personal_info_province', 'personal_info_marriage'], hash_bucket_size=2000),
	tf.feature_column.crossed_column(['personal_info_sex', age_buckets, 'personal_info_industry', 'personal_info_edu'], hash_bucket_size=2000),
	tf.feature_column.crossed_column(['personal_info_sex', age_buckets, 'devinfo_brand'], hash_bucket_size=2000),
	tf.feature_column.crossed_column([bucketized_operator_dial_0_6_num_per, bucketized_operator_dial_6_12_num_per, bucketized_operator_dial_12_18_num_per, bucketized_operator_dial_18_24_num_per], hash_bucket_size=2000),
    ]

    wide_columns = base_columns + crossed_columns

    deep_columns = [
        credit_card_num,
        credit_bank_num,
        credit_limit_avg,
        credit_limit_total,
        credit_limit_max,
        credit_limit_min,
        operator_time_to_auth,
        operator_time_to_2000,
        operator_duration_one_month,
        operator_dial_num_one_month,
        operator_dial_num_three_month,
        operator_num_one_month,
        operator_num_three_month,
        operator_duration_avg,
        operator_duration_stddev,
        operator_dial_duration_percent,
        operator_dial_num_percent,
        operator_dial_0_6_dur_per,
        operator_dial_0_6_num_per,
        operator_dial_6_12_dur_per,
        operator_dial_6_12_num_per,
        operator_dial_12_18_dur_per,
        operator_dial_12_18_num_per,
        operator_dial_18_24_dur_per,
        operator_dial_18_24_num_per,
        operator_dial_work_dur_per,
        operator_dial_work_num_per,
        debit_regionalmobility,
        debit_capacity,
        debit_transmonthcount,
        debit_avgamountmonthly,
        debit_avgcountmonthly,
        debit_diffday,
        debit_maxtransinterval,
	tf.feature_column.indicator_column(personal_info_sex),
	tf.feature_column.indicator_column(age_buckets),
	tf.feature_column.indicator_column(personal_info_industry),
	tf.feature_column.indicator_column(personal_info_edu),
	tf.feature_column.indicator_column(personal_info_marriage),
	tf.feature_column.indicator_column(devinfo_brand),
    ]

    return wide_columns, deep_columns


def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  hidden_units = [75, 50, 25]
  print(".........end  build_model_columns............")
  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}))

  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size):
  """Generate an input function for the Estimator."""
  assert tf.gfile.Exists(data_file), (
      '%s not found. Please make sure you have run data_download.py and '
      'set the --data_dir argument to the correct path.' % data_file)

  def parse_csv(value):
    print('Parsing', data_file)
    columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
    features = dict(zip(_CSV_COLUMNS, columns))
    # get lables from csv
    labels = features.pop('fpd30')
    return features, tf.equal(labels, '0')

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_file)

  if shuffle:
    dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

  dataset = dataset.map(parse_csv, num_parallel_calls=5)
  print('....................')
  print(dataset)
  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  return dataset


def main(argv):
  parser = WideDeepArgParser()
  flags = parser.parse_args(args=argv[1:])

  # Clean up the model directory if present
  # shutil.rmtree(flags.model_dir, ignore_errors=True)
  model = build_estimator(flags.model_dir, flags.model_type)

  train_file = os.path.join(flags.data_dir, _DIR_CONFIG['train_file'])
  test_file = os.path.join(flags.data_dir, _DIR_CONFIG['test_file'])

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  def train_input_fn():
    return input_fn(
        train_file, flags.epochs_between_evals, True, flags.batch_size)

  def eval_input_fn():
    return input_fn(test_file, 1, False, flags.batch_size)

  loss_prefix = LOSS_PREFIX.get(flags.model_type, '')
  train_hooks = hooks_helper.get_train_hooks(
      flags.hooks, batch_size=flags.batch_size,
      tensors_to_log={'average_loss': loss_prefix + 'head/truediv',
                      'loss': loss_prefix + 'head/weighted_loss/Sum'})

  # Train and evaluate the model every `flags.epochs_between_evals` epochs.
  for n in range(flags.train_epochs // flags.epochs_between_evals):
    model.train(input_fn=train_input_fn, hooks=train_hooks)
    results = model.evaluate(input_fn=eval_input_fn)

    # Display evaluation metrics
    print('Results at epoch', (n + 1) * flags.epochs_between_evals)
    print('-' * 60)

    for key in sorted(results):
      print('%s: %s' % (key, results[key]))

    if model_helpers.past_stop_threshold(
        flags.stop_threshold, results['accuracy']):
      break


class WideDeepArgParser(argparse.ArgumentParser):
  """Argument parser for running the wide deep model."""

  def __init__(self):
    super(WideDeepArgParser, self).__init__(parents=[
        parsers.BaseParser(multi_gpu=False, num_gpu=False)])
    self.add_argument(
        '--model_type', '-mt', type=str, default='wide_deep',
        choices=['wide', 'deep', 'wide_deep'],
        help='[default %(default)s] Valid model types: wide, deep, wide_deep.',
        metavar='<MT>')
    self.set_defaults(
        data_dir=_DIR_CONFIG['data_dir'],
        model_dir=_DIR_CONFIG['model_dir'],
        train_epochs=40,
        epochs_between_evals=2,
        batch_size=40)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  main(argv=sys.argv)
