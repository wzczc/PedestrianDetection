dataset = dict(
    sampling_function='kp_detection',
    batch_size=16,
    hyratio=0.5,
    size_train=(336, 448),
    size_test=(480, 640),
    img_channel_mean=[103.939, 116.779, 123.68])
train_cfg = dict(
    opt_algo="adam",
    prefetch_size=5,
    num_epochs=80,
    iter_per_epoch=2000,
    alpha=0.999,
    chunk_sizes=[8,8],
    sample_module='Caltech',
    learning_rate=1e-4,
    display=20,
    pretrain=None,
    cache_ped='data/cache/caltech/train_gt',
    cache_emp='data/cache/caltech/train_nogt',
    work_dir='./output')
test_cfg = dict(
    test=None,
    sample_module='Caltech',
    save_dir='./output/result',
    data_dir='./Caltech/test',
    nms_algorithm='exp_soft_nms',
    nms_threshold=0.5,
    scores_csp=0.01,
    scores_head=0.01)
backbone = dict(
    pretrained='torchvision://resnet50',
    depth=50,
    num_stages=4,
    dilations=(1, 1, 1, 2),
    strides=(1, 2, 2, 1),
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    norm_cfg=dict(type='BN', requires_grad=True),
    norm_eval=False,
    style='pytorch')
kp_head = dict(
    in_channels=[256, 512, 1024, 2048],
    out_channels=256,
    start_level=1,
    fusion_level=0,
    end_level=3,
    csp_center_loss=dict(
        beta=4,
        gamma=2,
        loss_weight=0.01),
    regr_h_loss=dict(
        loss_weight=1.0),
    regr_offset_loss=dict(
        loss_weight=0.1),
    head_center_loss=dict(
        beta=4,
        gamma=2,
        loss_weight=0.001),
    head_regr_loss=dict(
        loss_weight=0.1),
    head_offset_loss=dict(
        loss_weight=0.01))
    
