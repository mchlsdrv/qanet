import DataHandeling
import os
from datetime import datetime
import Networks as Nets
from RibCage import RibCage

__author__ = 'arbellea@post.bgu.ac.il'

# ROOT_SAVE_DIR = '/Users/aarbelle/Documents/DeepCellSegOut'
# ROOT_DATA_DIR3D = '/Users/aarbelle/Documents/CellTrackingChallenge/Training'


# ROOT_DATA_DIR = './data/Fluo-N2DH-SIM+/'
ROOT_DATA_DIR = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/BGUSIM/Fluo-N2DH-BGUSIM/Train/'
TEST_ROOT_DATA_DIR = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/BGUSIM/Fluo-N2DH-BGUSIM/Test/'
ROOT_SAVE_DIR = './output'
# ROOT_SAVE_DIR = '/newdisk/arbellea/LSTMUNet-tf2/'


class ParamsBase(object):

    def _override_params_(self, params_dict: dict):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        this_dict = self.__class__.__dict__.keys()
        for key, val in params_dict.items():
            if key not in this_dict:
                print('Warning!: Parameter:{} not in defualt parameters'.format(key))
            setattr(self, key, val)

    pass


class CTCParams(ParamsBase):
    # --------General-------------
    experiment_name = 'MyRun_SIM'
    dry_run = False  # Default False! Used for testing, when True no checkpoints or tensorboard outputs will be saved
    gpu_id = 2  # set -1 for CPU or GPU index for GPU.

    #  ------- Data -------
    data_provider_class = DataHandeling.CTCRAMReaderSequence2D
    root_data_dir = ROOT_DATA_DIR
    train_sequence_list = [('Fluo-N2DH-SIM+', '01'), ('Fluo-N2DH-SIM+', '02')]
    val_sequence_list = [('Fluo-N2DH-SIM+', '01'), ('Fluo-N2DH-SIM+', '02')]
    crop_size = (128, 128)
    batch_size = 5
    unroll_len = 4
    # data_format = 'NHWC'
    data_format = 'NCHW'
    train_q_capacity = 200
    val_q_capacity = 200
    num_val_threads = 2
    num_train_threads = 8

    # -------- Network Architecture ----------
    net_model = Nets.ULSTMnet2D
    net_kernel_params = {
        'down_conv_kernels': [
            [(5, 128), (5, 128)],
            [(5, 256), (5, 256)],
            [(5, 256), (5, 256)],
            [(5, 512), (5, 512)],
        ],
        'lstm_kernels': [
            [(5, 128)],
            [(5, 256)],
            [(5, 256)],
            [(5, 512)],
        ],
        'up_conv_kernels': [
            [(5, 256), (5, 256)],
            [(5, 128), (5, 128)],
            [(5, 64), (5, 64)],
            [(5, 32), (5, 32), (1, 3)],
        ],

    }

    # -------- Training ----------
    class_weights = [0.15, 0.25, 0.6]
    learning_rate = 1e-5
    num_iterations = 1000000
    validation_interval = 1000
    print_to_console_interval = 100

    # ---------Save and Restore ----------
    load_checkpoint = True
    load_checkpoint_path = '/newdisk/arbellea/LSTMUNet-tf2/LSTMUNet/MyRun_SIM/2019-05-06_115804'  # Used only if load_checkpoint is True
    continue_run = False
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 5000
    save_checkpoint_every_N_hours = 24
    save_checkpoint_max_to_keep = 5

    # ---------Tensorboard-------------
    tb_sub_folder = 'LSTMUNet'
    write_to_tb_interval = 500
    save_log_dir = ROOT_SAVE_DIR

    # ---------Debugging-------------
    profile = False

    def __init__(self, params_dict):
        self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.train_data_base_folders = [(os.path.join(ROOT_DATA_DIR, ds[0]), ds[1]) for ds in
                                        self.train_sequence_list]
        self.val_data_base_folders = [(os.path.join(ROOT_DATA_DIR, ds[0]), ds[1]) for ds in
                                      self.val_sequence_list]
        self.train_data_provider = self.data_provider_class(sequence_folder_list=self.train_data_base_folders,
                                                            image_crop_size=self.crop_size,
                                                            unroll_len=self.unroll_len,
                                                            deal_with_end=0,
                                                            batch_size=self.batch_size,
                                                            queue_capacity=self.train_q_capacity,
                                                            data_format=self.data_format,
                                                            randomize=True,
                                                            return_dist=False,
                                                            num_threads=self.num_train_threads
                                                            )
        self.val_data_provider = self.data_provider_class(sequence_folder_list=self.val_data_base_folders,
                                                          image_crop_size=self.crop_size,
                                                          unroll_len=self.unroll_len,
                                                          deal_with_end=0,
                                                          batch_size=self.batch_size,
                                                          queue_capacity=self.train_q_capacity,
                                                          data_format=self.data_format,
                                                          randomize=True,
                                                          return_dist=False,
                                                          num_threads=self.num_val_threads
                                                          )

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        base_path = None
        if self.load_checkpoint:
            if os.path.isdir(self.load_checkpoint_path) and not (self.load_checkpoint_path.endswith('tf_ckpts') or
                                                                 self.load_checkpoint_path.endswith('tf_ckpts/')):
                base_path = self.load_checkpoint_path
                self.load_checkpoint_path = os.path.join(self.load_checkpoint_path, 'tf_ckpts')
            elif os.path.isdir(self.load_checkpoint_path):
                base_path = os.path.dirname(self.load_checkpoint_path)
            else:

                base_path = os.path.dirname(os.path.dirname(self.load_checkpoint_path))
        if self.continue_run:
            self.experiment_log_dir = self.experiment_save_dir = base_path
        else:
            self.experiment_log_dir = os.path.join(self.save_log_dir, self.tb_sub_folder, self.experiment_name,
                                                   now_string)
            self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, self.experiment_name,
                                                    now_string)

        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if self.profile:
            if 'CUPTI' not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/extras/CUPTI/lib64/'


class CTCInferenceParams(ParamsBase):
    gpu_id = 3  # for CPU ise -1 otherwise gpu id
    # model_path = '/newdisk/arbellea/LSTMUNet-tf2/LSTMUNet/MyRun_SIM/2019-05-06_115804'
    ts = '2021-11-15_182833'
    model_path = f'./output/QANet/SIM+/{ts}'
    output_path = f'./output/QANet/SIM+/{ts}/inference'
    sequence_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Test/Fluo-N2DH-SIM+/01/'
    filename_format = 't*.tif'  # default format for CTC

    data_reader = DataHandeling.CTCInferenceReader
    FOV = 0
    # data_format = 'NHWC'  # 'NCHW' or 'NHWC'
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'
    min_cell_size = 10
    max_cell_size = 100
    edge_dist = 5
    pre_sequence_frames = 0

    # ---------Debugging---------
    dry_run = True
    save_intermediate = True
    save_intermediate_path = ''
    save_intermediate_label_path = ''

    def __init__(self, params_dict: dict = None):
        if params_dict is not None:
            self._override_params_(params_dict)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path, now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)


class QANetParams(ParamsBase):
    # --------General-------------
    experiment_name = 'SIM+'
    dry_run = False  # Default False! Used for testing, when True no checkpoints or tensorboard outputs will be saved
    gpu_id = 3  # set -1 for CPU or GPU index for GPU.

    #  ------- Data -------
    data_provider_class = DataHandeling.CTCRAMReaderQANet2D
    root_data_dir = ROOT_DATA_DIR
    # train_sequence_list = [('Fluo-N2DH-MYSIMV3', '{:02d}'.format(s+1)) for s in range(46)]
    # val_sequence_list = [('Fluo-N2DH-MYSIMV3', '{:02d}'.format(s+1)) for s in range(46, 51)]
    # val_perscent = 0.
    train_sequence_list = [('', '{:02d}'.format(s+1)) for s in range(44)]
    # train_sequence_list = [('Fluo-N2DH-SIM+', '{:02d}'.format(s+1)) for s in range(44)]
    val_sequence_list = [('', '{:02d}'.format(s+1)) for s in range(44)]
    # val_sequence_list = [('Fluo-N2DH-SIM+', '{:02d}'.format(s+1)) for s in range(2)]
    val_perscent = 0.2
    labeled_seg = False
    crop_size = (256, 256)
    batch_size = 32
    # data_format = 'NHWC'
    data_format = 'NCHW'
    train_q_capacity = 1000
    val_q_capacity = 200
    num_val_threads = 2
    num_train_threads = 32

    # -------- Network Architecture ----------
    net_model = RibCage
    net_kernel_params = {
        'ribcage': [(5, 32),
                    (5, 64),
                    (5, 128),
                    (5, 256)],
        'fc': [512, 1024, 1],
        'drop_rate': 0.2
    }
    net_kernel_params = {
        'ribcage': [[3, 64],
                    [3, 128],
                    [3, 256],
                    [3, 256]],
        'fc': [128, 128, 1],
        'drop_rate': 0.0
    }
    # -------- Training ----------
    learning_rate = 1e-3
    num_iterations = 1000000
    validation_interval = 500
    print_to_console_interval = 100

    # ---------Save and Restore ----------
    load_checkpoint = False
    load_checkpoint_path = '/newdisk/arbellea/LSTMUNet-tf2/QANet/FixLossTest/2019-05-28_105824/'  # Used only if load_checkpoint is True
    continue_run = False
    save_checkpoint_dir = ROOT_SAVE_DIR
    save_checkpoint_iteration = 5000
    save_checkpoint_every_N_hours = 24
    save_checkpoint_max_to_keep = 5

    # ---------Tensorboard-------------
    tb_sub_folder = 'QANet'
    write_to_tb_interval = 500
    save_log_dir = ROOT_SAVE_DIR

    # ---------Debugging-------------
    profile = False

    def __init__(self, params_dict):
        self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.train_data_base_folders = [(os.path.join(ROOT_DATA_DIR, ds[0]), ds[1]) for ds in
                                        self.train_sequence_list]
        self.val_data_base_folders = [(os.path.join(ROOT_DATA_DIR, ds[0]), ds[1]) for ds in
                                      self.val_sequence_list]
        self.train_data_provider = self.data_provider_class(sequence_folder_list=self.train_data_base_folders,
                                                            image_crop_size=self.crop_size,
                                                            batch_size=self.batch_size,
                                                            queue_capacity=self.train_q_capacity,
                                                            data_format=self.data_format,
                                                            num_threads=self.num_train_threads,
                                                            dataset_percent=1-self.val_perscent,
                                                            labeled_seg=self.labeled_seg
                                                            )
        self.val_data_provider = self.data_provider_class(sequence_folder_list=self.val_data_base_folders,
                                                          image_crop_size=self.crop_size,
                                                          batch_size=self.batch_size,
                                                          queue_capacity=self.train_q_capacity,
                                                          data_format=self.data_format,
                                                          num_threads=self.num_val_threads,
                                                          dataset_percent=-self.val_perscent,
                                                          labeled_seg=self.labeled_seg
                                                          )

        now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
        base_path = None
        if self.load_checkpoint:
            if os.path.isdir(self.load_checkpoint_path) and not (self.load_checkpoint_path.endswith('tf_ckpts') or
                                                                 self.load_checkpoint_path.endswith('tf_ckpts/')):
                base_path = self.load_checkpoint_path
                self.load_checkpoint_path = os.path.join(self.load_checkpoint_path, 'tf_ckpts')
            elif os.path.isdir(self.load_checkpoint_path):
                base_path = os.path.dirname(self.load_checkpoint_path)
            else:

                base_path = os.path.dirname(os.path.dirname(self.load_checkpoint_path))
        if self.continue_run:
            self.experiment_log_dir = self.experiment_save_dir = base_path
        else:
            self.experiment_log_dir = os.path.join(self.save_log_dir, self.tb_sub_folder, self.experiment_name,
                                                   now_string)
            self.experiment_save_dir = os.path.join(self.save_checkpoint_dir, self.tb_sub_folder, self.experiment_name,
                                                    now_string)

        if not self.dry_run:
            os.makedirs(self.experiment_log_dir, exist_ok=True)
            os.makedirs(self.experiment_save_dir, exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'train'), exist_ok=True)
            os.makedirs(os.path.join(self.experiment_log_dir, 'val'), exist_ok=True)
        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if self.profile:
            if 'CUPTI' not in os.environ['LD_LIBRARY_PATH']:
                os.environ['LD_LIBRARY_PATH'] += ':/usr/local/cuda/extras/CUPTI/lib64/'


class QANetInferenceParams(ParamsBase):
    gpu_id = 1  # for CPU ise -1 otherwise gpu id

    # model_path = '/newdisk/arbellea/LSTMUNet-tf2/QANet/FixLossTest/2019-05-28_105824/'
    ts = '2021-11-15_182833'
    model_path = f'./output/QANet/SIM+/{ts}'
    output_path = f'./output/QANet/SIM+/{ts}/inference'
    sequence_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N2DH-SIM+/01/'
    sequence_seg_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N2DH-SIM+/01_RES/'
    sequence_gt_path = '/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Fluo-N2DH-SIM+/01_GT/SEG'
    filename_format = 't*.tif'  # default format for CTC
    filename_seg_format = 'mask*.tif'  # default format for CTC
    filename_gt_format = 'man_seg*.tif'  # default format for CTC
    root_data_dir = ROOT_DATA_DIR
    data_reader = DataHandeling.CTCInferenceReader
    FOV = 0
    # data_format = 'NHWC'  # 'NCHW' or 'NHWC'
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'

    # ---------Debugging---------

    dry_run = True
    save_intermediate = False
    save_intermediate_path = ''

    def __init__(self, params_dict: dict = None):
        if params_dict is not None:
            self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path, now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)


class QANetCTCTestParams(ParamsBase):
    gpu_id = 1  # for CPU ise -1 otherwise gpu id
    model_path = './output/QANet/SIM+/2022-01-01_083759'
    # model_path = './output/QANet/SIM+/2021-11-15_182833'
    # model_path = '/newdisk/arbellea/LSTMUNet-tf2/QANet/SIM+/2019-10-04_231842/'
    # model_path = '/newdisk/arbellea/LSTMUNet-tf2/QANet/MySIMV3/2019-08-07_080206'
    # override_checkpoint = '/newdisk/arbellea/LSTMUNet-tf2/QANet/MySIMV3/2019-08-07_080206/tf_ckpts/ckpt-1000000'
    override_checkpoint = False
    output_path = './tmp/output'
    # data_path = '/media/rrtammyfs/Users/arbellea/QANet/TestData/SIM+'
    # data_params = {'data_path': '/media/rrtammyfs/Users/arbellea/QANet/TestData/SIM+'}
    data_params = {'data_path': '/media/rrtammyfs/Users/arbellea/QANet/TestData/SIM+V2'}
    test_sequence_list = [('', '{:02d}'.format(s+1)) for s in range(45, 51)]
    # data_params = {'data_path': '/media/rrtammyfs/Users/arbellea/QANet/TestData/MySIMV3'}

    # TODO: The data_reader changed
    data_reader = DataHandeling.CTCTestQANetReader

    data_provider_class = DataHandeling.CTCRAMReaderQANet2D
    # data_params = {'data_path': '/media/rrtammyfs/labDatabase/CellTrackingChallenge/BGUSIM/Fluo-N2DH-BGUSIM/Test/'}
    test_data_base_folders = [
        (os.path.join(TEST_ROOT_DATA_DIR, ds[0]), ds[1]) for ds in test_sequence_list
    ]
    self.test_data_provider = data_provider_class(
        sequence_folder_list=test_data_base_folders,
        image_crop_size=(256, 256),
        batch_size=32,
        queue_capacity=1000,
        num_threads=32,
    )

    has_gt = True
    FOV = 0
    # data_format = 'NHWC'  # 'NCHW' or 'NHWC'
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'
    bias=0.06094262067973614

    # ---------Debugging---------

    dry_run = False
    save_intermediate = False
    save_intermediate_path = ''

    def __init__(self, params_dict: dict = None):
        if params_dict is not None:
            self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path, now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)


class QANetLSCTestParams(ParamsBase):
    gpu_id = 1  # for CPU ise -1 otherwise gpu id
    model_path = '/newdisk/arbellea/LSTMUNet-tf2/QANet/LSC2017_BestDiceFlip/2019-09-23_142936/'
    # bias = 0.004725349456071853
    # model_path = '/newdisk/arbellea/LSTMUNet-tf2/QANet/LSC2017_BestDice_Fast/2019-09-27_210424/'
    # bias = 0.009394581139087676
    override_checkpoint = False
    output_path = './tmp/output'
    images_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/CVPPP2017_LSC_training_data_validation2/'
    segmetation_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/CVPPP2017_LSC_training_data_validation2/'
    score_file = ('/media/rrtammyfs/labDatabase/LeafSegmentation/CVPPP2017_LSC_training_data_validation2/'
                  'scores_per_image.txt')
    bias = 0
    # bias =  0.0060885860919952395
    images_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/CVPPP2017_testing_images/PNG'

    segmetation_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/WardResults/CVPPP2017Test/PNG'
    score_file = ('/media/rrtammyfs/labDatabase/LeafSegmentation/WardResults/CVPPP2017Test/output_file/'
                  'scores_per_image.txt')
    #
    # segmetation_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/answer/PNG/'
    # score_file = ('/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/output_file_final/'
    #               'scores_per_image.txt')
    #
    # segmetation_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/corrupted/PNG/'
    # score_file = ('/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/corrupted/output_file/'
    #               'scores_per_image.txt')
    #
    # segmetation_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/corrupted2/PNG/'
    # score_file = '/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/corrupted2/output_file/scores_per_image.txt'

    # segmetation_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/corrupted3/PNG/'
    # score_file = '/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/corrupted3/output_file/scores_per_image.txt'

    # segmetation_path = '/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/corrupted4/PNG/'
    # score_file = '/media/rrtammyfs/labDatabase/LeafSegmentation/RonnyKimmelResults/corrupted4/output_file/scores_per_image.txt'

    data_params = {'data_path': images_path,
                   'segmetation_path': segmetation_path,
                   'score_file': score_file,
                   'crop_shape': (432, 432)}
    data_reader = DataHandeling.LSCTestQANetReader
    has_gt = True
    FOV = 0
    # data_format = 'NHWC'  # 'NCHW' or 'NHWC'
    data_format = 'NCHW'  # 'NCHW' or 'NHWC'

    # ---------Debugging---------

    dry_run = False
    save_intermediate = False
    save_intermediate_path = ''

    def __init__(self, params_dict: dict = None):
        if params_dict is not None:
            self._override_params_(params_dict)
        if isinstance(self.gpu_id, list):

            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)[1:-1]
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(self.gpu_id)

        self.channel_axis = 1 if self.data_format == 'NCHW' else 3
        if not self.dry_run:
            os.makedirs(self.output_path, exist_ok=True)
            if self.save_intermediate:
                now_string = datetime.now().strftime('%Y-%m-%d_%H%M%S')
                self.save_intermediate_path = os.path.join(self.save_intermediate_path, now_string)
                self.save_intermediate_vis_path = os.path.join(self.save_intermediate_path, 'Softmax')
                self.save_intermediate_label_path = os.path.join(self.save_intermediate_path, 'Labels')
                os.makedirs(self.save_intermediate_path, exist_ok=True)
                os.makedirs(self.save_intermediate_vis_path, exist_ok=True)
                os.makedirs(self.save_intermediate_label_path, exist_ok=True)
