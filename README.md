# Evolutionary Algorithms and pruning: Comparison of three after-prune training methods

In this project we use evolutionary algorithms to find pruned networks out of feed forward neural networks that improve their overall performance. Furthermore, we test three different approaches regarding the training of the pruned network, i.e. no training, retrain the network or re-instantiate a new network with its connections pruned.

For further details please read the Applied Research Project [report](<https://github.com/martingorosito/EA_Pruning_After_Prune_Training/blob/main/ARP%20Report/Gorosito%2C%20Martin%20(26567)%20ARP.pdf>)

## Description

We use six different datasets and implement a neural network for each of them. Then we use binary strings to represent the weight matrices, and use them as individuals for the evolutionary search. The ones and zeros represent whether or not the weight is present or pruned respectively. We run three different algorithms with the same characteristics, except the fitness function, whcih is valued according to the after-prune method being tested: "No Training" (NT), "Train after Prune" (TAP) and "Train from Scratch" (TFS). Each EA search was run 10 times with different seeds.

### Datasets and network characteristics

The datasets we use can be obtained from either the UCI repository or Kaggle. Their characteristics are summarized in the table below, along with the feed forward networks layers sizes and the number of epochs we use to train them. These were obtained in the paper by Cantu-Paz and Kamath linked below. 

|Dataset|Examples|Classes|Input|Hidden|Output|Epochs|
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
|[Iris](<https://archive.ics.uci.edu/ml/datasets/iris>)|150|3|4|5|3|80|
|[P-Diabetes](<https://www.kaggle.com/uciml/pima-indians-diabetes-database>)|768|2|8|5|2|30|
|[Breast-W](<https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29>)|699|2|9|5|2|20|
|[Wine](<https://archive.ics.uci.edu/ml/datasets/wine>)|178|3|13|5|3|15|
|[Ionosphere](<https://archive.ics.uci.edu/ml/datasets/ionosphere>)|351|2|34|10|2|40|
|[Sonar](<http://archive.ics.uci.edu/ml/datasets/connectionist+bench+(sonar,+mines+vs.+rocks)>)|208|2|60|10|2|60|

### Evolutionary Algorithm
Before the algorithm begins, the datasets are divided into a training set and a test set. The test set is separated and used later in the evaluation section. The training set is subdivided into a sub-trianing set and a validation set. 

Each individual is represented as a binary string, which is a concatenation of the three weight matrices. These can then be reshaped into a pruning mask.
The algorithms have a benchmark individual, which is the feed forward network with all of its connections present. This individual is trained on the sub-training set and tested on the validation set. This is the network thas is pruned through the search. 

For reproduction, we use three individual tournament selection with two winners being selected to be parents. These generate one offspring using uniform crossover. The offspring is mutated using bitflip.

The fitness functions is where the algorithms differ. For the "No Training" method, the individual is pruned and tested on the validation set. The accuracy of the individual is used as its fitness value. 
For the "Train from Scratch" method, a new network is instantiated, pruned and trained using the sub-training set. The accuracy on the validation set is used as a fitness value. 
For the "Train after Prune" mehtod, the network is pruned, then retrained some more on the sub-training set. The accuracy on the valdation set is used as a fitness value. 

The individuals found are all kept on a history list. Then those that share the same fitness with the best individuals found are kept as results of the EA search. Finally, each of these are evaluated using combined 5x2 cv F Test. 

## Results
The EA search show that the "No Training" method produces no improved networks, with significant deterioration to their accuracy on test data. 

As for the "Train from Scratch" and the "Train after Prune" methods, their results are quite similar, both being able to produce improved networks. To compare these we use the following formula:

EO = PAN ( PBI * AI + (1 - PBI) * AD)

with:
- EO = Expected Output
- PAN = Percentage of Accepted Networks. These are the networks that passed the 5x2 cv F test
- PBI = Percentage of Better Individuals. The accepted networks that improve on the benchmark individual
- AI = Average improvement
- AD = Average deterioration

The results for this formula can be summarized in the following graph:

![image](https://user-images.githubusercontent.com/29287072/158018131-271f9386-f72f-4b08-977a-7094d359e684.png)

This shows how larger networks are more suitable for pruning using EA, as the networks used for the Sonar and Ionosphere datasets are the ones that most benefit from it. Nonetheless, the project did not provide a definitive answer as to which method is the best, but rather gave us an idea to it. Furthermore, it was noted that since the initial individuals are generated using uniformly random distribution, that the density of weights would average at 50%, which is a lot to prune from a network. 

### Links
[Cantú-Paz, E., & Kamath, C. (2005). An empirical comparison of combinations of evolutionary algorithms and neural networks for classification problems](<https://ieeexplore.ieee.org/document/1510768>)

[Alpaydin, E. (1999). Combined 5x2 cv F Test for Comparing Supervised Classification Learning Algorithms.](<https://direct.mit.edu/neco/article-abstract/11/8/1885/6310/Combined-5-2-cv-F-Test-for-Comparing-Supervised?redirectedFrom=fulltext>)
