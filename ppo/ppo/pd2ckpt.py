import paddle

params_dict = paddle.load("E:/20212333Python\demo05_paddle\ppo\ppo/train_log\model.pdparams")
paddle.save(params_dict, "model1.ckpt")