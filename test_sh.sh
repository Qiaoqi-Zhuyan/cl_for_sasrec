python main.py --device=cuda \
--dataset=user_fix \
--train_dir=default \
--state_dict_path='user_fix_default/CLS_SASRec.epoch=701.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.batch_size=256_temp=1.0.pth' \
--inference_only=true \
--maxlen=50 \
--topk=10

# ---- cube
#SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=10.pth
#SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.batch_size=128.pth
#SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.batch_size=256.pth
#SASRec.epoch=601.lr=0.001.layer=2.head=1.hidden=50.maxlen=50.pth
#CLS_SASRec.epoch=701.lr=0.0005.layer=2.head=1.hidden=50.maxlen=50.batch_size=256_temp=0.7.pth
# SASRec.epoch=701.lr=0.0005.layer=2.head=1.hidden=50.maxlen=50.batch_size=256.pth