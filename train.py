import wandb
from tqdm import tqdm
from models.models_operate import *
from data.image_folder import ImageFolder
from util.options import get_args_parser, print_options

import util.lr_decay as lrd
import timm.optim.optim_factory as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# get and print options
parser = get_args_parser()
args = parser.parse_args()
print_options(args, parser)

# dataset construct

dataset_train = ImageFolder(args, is_train=True)
dataset_test = ImageFolder(args, is_train=False)
# train_size = int(0.8 * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
sampler_train = torch.utils.data.RandomSampler(data_source=dataset_train, replacement=True,
                                               num_samples=int(len(dataset_train)))
sampler_test = torch.utils.data.SequentialSampler(data_source=dataset_test)

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    pin_memory=args.pin_mem,
    drop_last=True,
)
data_loader_test = torch.utils.data.DataLoader(dataset_test, sampler=sampler_test)

# Model loaded
chkpt_dir = r'weights/mae_visualize_vit_large_ganloss.pth'
model = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')
print('Dataset loaded. Train dataset {}. Test dataset {}.'.format(len(dataset_train), len(dataset_test)))
model.to(device)

eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

if args.lr is None:  # only base_lr is specified
    args.lr = args.blr * eff_batch_size / 256

print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
print("actual lr: %.2e" % args.lr)

print("accumulate grad iterations: %d" % args.accum_iter)
print("effective batch size: %d" % eff_batch_size)
# following timm: set wd as 0 for bias and norm layers
# param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
# optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
print(optimizer)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
loss_scaler = NativeScaler()
torch.manual_seed(2)
accum_iter = args.accum_iter
if args.wandb_flag:
    wandb.init(project="iml", entity="chelton")
    # 断点续训
    # wandb.init(project="iml", entity="chelton", resume=True)
    # wandb.config = {'args': args}
    # id = wandb.util.generate_id()
    # wandb.init(id=id, resume="allow")
    # # or via environment variables
    # os.environ["WANDB_RESUME"] = "allow"
    # os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()
    wandb.init()
for epoch in tqdm(range(args.start_epoch, args.epochs)):
    model.train()
    epoch_loss = 0.0
    for i, (imgs, masks, _) in enumerate(data_loader_train):
        optimizer.zero_grad()
        imgs, masks = imgs.to(device), masks.to(device)
        loss, y, mask, patch_label, label_pred = model(imgs, masks, mask_ratio=0.75)

        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(i + 1) % accum_iter == 0)
        epoch_loss += loss.item()
        print("\rEpoch {}/{}. Batch number of {}. Learning rate is {:.2f}.".format(epoch, args.epochs - 1, i, optimizer.state_dict()['param_groups'][0]['lr']), flush=True, end='')

    # scheduler.step()
    if args.wandb_flag:
        wandb.log({"loss": epoch_loss, "lr":optimizer.state_dict()['param_groups'][0]['lr']})

    if args.output_dir and ((epoch + 1) % 50 == 0 or epoch + 1 == args.epochs):
        # save checkpoints
        misc.save_model(
            args=args, model=model, model_without_ddp=model, optimizer=optimizer,
            loss_scaler=loss_scaler, epoch=epoch)

        # test phase
        model.eval()
        print()
        test_model(model=model, dataset=dataset_test, data_loader=data_loader_test)
