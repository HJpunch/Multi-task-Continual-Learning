task_mapping = {
  'llcm': {
    'loss_weight': 1.0,
    'gres_ratio': 1,
    'task_name': 'DataBuilder_cross',
    'test_task_type': 'cross',
    'root_path': '../Instruct-ReID/data/llcm',
    'train_file_path': '../Instruct-ReID/data/llcm/train.txt',
    'query_list': '../Instruct-ReID/data/llcm/query.txt',
    'gallery_list': '../Instruct-ReID/data/llcm/gallery.txt',
    'dataset_name': 'LLCM' 
  },

  'ltcc': {
    'loss_weight': 1.0,
    'gres_ratio': 1,
    'task_name': 'DataBuilder_cc',
    'test_task_type': 'cc',
    'root_path': '../Instruct-ReID/data/ltcc',
    'train_file_path': '../Instruct-ReID/data/ltcc/datalist/train.txt',
    'query_list': '../Instruct-ReID/data/ltcc/datalist/query_cc.txt',
    'gallery_list': '../Instruct-ReID/data/ltcc/datalist/gallery_cc.txt',
    'dataset_name': 'LTCC'
  },

  'market': {
    'loss_weight': 1.0,
    'gres_ratio': 1,
    'task_name': 'DataBuilder_sc',
    'test_task_type': 'sc',
    'root_path': '../Instruct-ReID/data/market',
    'train_file_path': '../Instruct-ReID/data/market/datalist/train.txt',
    'query_list': '../Instruct-ReID/data/market/datalist/query.txt',
    'gallery_list': '../Instruct-ReID/data/market/datalist/gallery.txt',
    'dataset_name': 'Market1501'
  },

  'cuhk_pedes': {
    'loss_weight': 1.0,
    'gres_ratio': 1,
    'task_name': 'DataBuilder_t2i',
    'attt_file': '../Instruct-ReID/data/cuhk_pedes/caption_t2i_v2.json',
    'root_path': '../Instruct-ReID/data/cuhk_pedes',
    'train_file_path': '../Instruct-ReID/data/cuhk_pedes/train_t2i_v2.txt',
  }
}