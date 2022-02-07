# Virtual data augmentation method for reaction prediction in small dataset scenario
This is the code for "Virtual data augmentation method for reaction prediction in small dataset scenario" paper.  

#Setup

```bash
conda env create -n ABCD -f environment.yml
conda activate ABCD
```

# Dataset
USPTO_41W was used as pretraining dataset: `data/41W_pretrain`
The data for training, dev and testing of Buchwald-Hartwig are provided in ```data/Buchwald-Hartwig``` file. 
The data for training, dev and testing of Chan-Lam are provided in ```data/Chan-Lam``` file.
The data for training, dev and testing of Hiyama are provided in```data/ Hiyama``` file.
The data for training, dev and testing of Kumada are provided in```data/ Kumada``` file.
The data for training, dev and testing of Suzuki are provided in```data/ Suzuki``` file.
The data for training, dev and testing of Suzuki_ratio are provided in```data/ Suzuki_ratio``` file.
The molecular Reaxys data can be found http://www.elsevier.com/online-tools/reaxys


# Quickstart
# Step 1: train  the model in different reaction prediction
run transformer.py 
get a baseline model


# Step 2. test
run test.py

# Step 3. Compare the accuracy of each reaction before and after augmenting the data set.
run Compare target and predicted accuracy.py

# Step 4: train the pretrained model in different reaction prediction 
run transformer.py 
get a transformer-transfer model
Continue Step 2 and 3, and get the accuracy of each reaction after applying the transfer learning strategy.