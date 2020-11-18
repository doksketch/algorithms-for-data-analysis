import numpy as np

class RandomForestClassifier:

    def __init__(
        self,
        max_depth: int=6,
        num_leaves: int=32,
        min_samples_leaf: int=5,
        criterion: str='gini',
        oob_score: bool = False,
        n_trees: int=10):

        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_samples_leaf = min_samples_leaf
        self.oob_score = oob_score
        self.n_trees = n_trees

        if criterion not in ["gini"]:
            raise ValueError(
                "Undefined criterion value. Please use: 'gini'."
            )

        self.information_criterio = self.gini if criterion == 'gini'
        self.current_leaves = 0
        self.current_depth = 0
    )


    # Генерация бустрэп выборок и подмножества признаков для нахождения разбиения в узле
    def get_bootstrap(self, data, labels, N):
        n_samples = data.shape[0]
        bootstrap = []
        is_oob_samples = np.zeros((data.shape[0], N))
        
        for i in range(N):
            b_data = np.zeros(data.shape)
            b_labels = np.zeros(labels.shape)
            used_samples = [] #для oob
            
            for j in range(n_samples):
                sample_index = random.randint(0, n_samples-1)
                b_data[j] = data[sample_index]
                b_labels[j] = labels[sample_index]
                used_samples.append(sample_index)
                
            for sample in used_samples:
                if sample not in used_samples:
                    is_oob_samples[i, j] == True
                else:
                    is_oob_samples[i, j] == False
                    bootstrap.append((b_data, b_labels))
        
        return bootstrap, is_oob_samples

    def get_subsample(self, len_sample):
        sample_indexes = [i for i in range(len_sample)]

        len_subsample = int(np.sqrt(len_sample))
        subsample = []

        np.random.shuffle(sample_indexes)

        for i in range(len_subsample):
            subsample.append(sample_indexes.pop())

        return subsample

    def gini(self, labels):
        
        classes = {}
        for label in labels:
            if label not in classes:
                classes[label] = 0
            classes[label] += 1
        
        
        impurity = 1
        for label in classes:
            p = classes[label] / len(labels)
            impurity -= p ** 2
            
        return impurity

    def quality(self, left_labels, right_labels, current_gini):

        p = float(left_labels.shape[0]) / (left_labels.shape[0] + right_labels.shape[0])
        
        return current_gini - p * gini(left_labels) - (1 - p) * gini(right_labels)

    def split(self, data, labels, index, t):
        
        left = np.where(data[:, index] <= t)
        right = np.where(data[:, index] > t)
            
        true_data = data[left]
        false_data = data[right]
        true_labels = labels[left]
        false_labels = labels[right]
            
        return true_data, false_data, true_labels, false_labels

    def find_best_split(self, data, labels):

        current_quality = self.information_criterio(labels)

        best_quality = 0
        best_t = None
        best_index = None
        
        n_features = data.shape[1]
        
        # выбор индекса из подвыборки длиной sqrt(n_features)
        subsample = get_subsample(n_features)
        
        for index in subsample:
            
            t_values = np.unique([row[index] for row in data])
            
            for t in t_values:
                true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)
                
                if len(true_data) < self.min_samples_leaf or len(false_data) < self.min_samples_leaf:
                    continue
                
                current_quality = quality(true_labels, false_labels, current_gini)
                
                
                if current_quality > best_quality:
                    best_quality, best_t, best_index = current_quality, t, index

        return best_quality, best_t, best_index 
    
    def build_tree(self, data, labels):
        
        quality, t, index = self.find_best_split(data, labels)
        
        if self.current_leaves >= self.num_leaves:
            return Leaf(X, y)

        if self.current_depth >= self.max_depth:
            return Leaf(X, y)

        if quality == 0:
            return Leaf(X, y)

        true_data, false_data, true_labels, false_labels = self.split(data, labels, index, t)

        true_branch = self.build_tree(true_data, true_labels)
        false_branch = self.build_tree(false_data, false_labels)

        self.current_depth += 1
        self.num_leaves += 2

        return Node(index, t, true_branch, false_branch)

    def oob_accuracy(data, labels, is_oob_sample, forest):
        correct = 0
        total = 0
        
        for obj in range(data.shape[0]):
            votes = []
            
            for tree in forest:
                if is_oob_sample[obj, tree]:
                    votes.append(classify_object(i,j))
                
            predicted = max(set(votes), key=votes.count)
            
            actual = labels[i]
            total += 1
            
            if actual == predicted:
                correct += 1
                
        oob_accuracy = correct / total
                
        return oob_accuracy
    
    def random_forest(self, data, labels, n_trees, oob_score):
        forest=[]
        bootstrap = self.get_bootstrap(data, labels, n_trees)

        for b_data, b_labels in bootstrap:
            forest.append(build_tree(b_data, b_labels))

        score = self.oob_accuracy(data, labels, is_oob_sample, forest) if oob_score

        return forest, score
    
    def classify_object(obj, node):

        if isinstance(node, Leaf):
            answer = node.prediction
            return answer

        if obj[node.index] <= node.t:
            return classify_object(obj, node.true_branch)
        else:
            return classify_object(obj, node.false_branch)
    
    
    def predict(self, data, tree):
    
        classes = []
        for obj in data:
            prediction = self.classify_object(obj, tree)
            classes.append(prediction)
    
        return classes

    #Предсказание голосованием деревьев
    def tree_vote(self, forest, data):
        predictions = []
        for tree in forest:
            predictions.append(predict(data, tree))

        #список с предсказаниями для каждого объекта
        predictions_per_object = list(zip(*predictions))

        # выберем в качестве итогового предсказания для каждого объекта то,
        # за которое проголосовало большинство деревьев
        voted_predictions = []
        for obj in predictions_per_object:
            voted_predictions.append(np.max(set(obj), key = obj.count))

        return voted_predictions

