# ablation on capacity
python train.py --dest models/ --epochs 50 --episodes 50 --capacity 500000
python train.py --dest models/ --epochs 50 --episodes 50 --capacity 50000
python train.py --dest models/ --epochs 50 --episodes 50 --capacity 5000

# ablation on sync-rate
python train.py --dest models/ --epochs 50 --episodes 50 --sync-rate 1
python train.py --dest models/ --epochs 50 --episodes 50 --sync-rate 10
python train.py --dest models/ --epochs 50 --episodes 50 --sync-rate 100
