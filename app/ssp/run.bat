@echo off

echo "GCN"

echo "Cora"
echo "===="

python experiments/gcn.py --dataset=Cora --split=public --optimizer=Adam --logger=GCN-Cora1-Adam
python experiments/gcn.py --dataset=Cora --split=public --optimizer=Adam --preconditioner=KFAC --logger=GCN-Cora1-Adam-KFAC

python experiments/gcn.py --dataset=Cora --split=public --optimizer=SGD --logger=GCN-Cora1-SGD
python experiments/gcn.py --dataset=Cora --split=public --optimizer=SGD --preconditioner=KFAC --logger=GCN-Cora1-SGD-KFAC

echo "CiteSeer"
echo "========"

python experiments/gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --logger=GCN-CiteSeer1-Adam
python experiments/gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --preconditioner=KFAC --logger=GCN-CiteSeer1-Adam-KFAC

python experiments/gcn.py --dataset=CiteSeer --split=public --optimizer=SGD --logger=GCN-CiteSeer1-SGD
python experiments/gcn.py --dataset=CiteSeer --split=public --optimizer=SGD --preconditioner=KFAC --logger=GCN-CiteSeer1-SGD-KFAC

echo "PubMed"
echo "======"

python experiments/gcn.py --dataset=PubMed --split=public --optimizer=Adam --logger=GCN-PubMed1-Adam
python experiments/gcn.py --dataset=PubMed --split=public --optimizer=Adam --preconditioner=KFAC --logger=GCN-PubMed1-Adam-KFAC

python experiments/gcn.py --dataset=PubMed --split=public --optimizer=SGD --logger=GCN-PubMed1-SGD
python experiments/gcn.py --dataset=PubMed --split=public --optimizer=SGD --preconditioner=KFAC --logger=GCN-PubMed1-SGD-KFAC



echo "Precondition-at experiments"
echo "======"

python experiments/gcn.py --dataset=Cora --split=public --optimizer=Adam --preconditioner=KFAC --logger=GCN-Cora1-Adam-KFAC --precondition_at=150
python experiments/gcn.py --dataset=Cora --split=public --optimizer=SGD --preconditioner=KFAC --logger=GCN-Cora1-SGD-KFAC --precondition_at=150

python experiments/gcn.py --dataset=CiteSeer --split=public --optimizer=Adam --preconditioner=KFAC --logger=GCN-CiteSeer1-Adam-KFAC --precondition_at=150
python experiments/gcn.py --dataset=CiteSeer --split=public --optimizer=SGD --preconditioner=KFAC --logger=GCN-CiteSeer1-SGD-KFAC --precondition_at=150

python experiments/gcn.py --dataset=PubMed --split=public --optimizer=Adam --preconditioner=KFAC --logger=GCN-PubMed1-Adam-KFAC --precondition_at=150
python experiments/gcn.py --dataset=PubMed --split=public --optimizer=SGD --preconditioner=KFAC --logger=GCN-PubMed1-SGD-KFAC --precondition_at=150
