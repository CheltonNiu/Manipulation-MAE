from models.models_operate import *
from data.image_folder import ImageFolder
from util.options import get_args_parser, print_options


# get and print options
parser = get_args_parser()
args = parser.parse_args()
print_options(args, parser)


# dataset construct
dataset_test = ImageFolder(args, is_train=False)
sampler_test = torch.utils.data.SequentialSampler(data_source=dataset_test)
data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test)


# Model loaded
chkpt_dir = r'checkpoints'
chkpt_list = os.listdir(chkpt_dir)
print(chkpt_list)

for chkpt in chkpt_list:
    if '-99.pth' in chkpt:
        chkpt_path = os.path.join(chkpt_dir, chkpt)
        model = prepare_model(chkpt_path, 'mae_vit_large_patch16')
        print('Model loaded.')
        print('Dataset loaded. Test dataset {}.'.format(len(dataset_test)))
        model.to(device)

        torch.manual_seed(2)

        # test phase
        model.eval()
        # test_model(model=model, dataset=dataset_test, data_loader=data_loader_test)
        visualize_model(model=model, dataset=dataset_test, data_loader=data_loader_test)