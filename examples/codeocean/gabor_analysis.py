# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score

# Assume we have pre-extracted activation maps and corresponding labels
# early_layer_activations and late_layer_activations are numpy arrays of shape (num_samples, num_features)
# gabor_orientations and class_labels are the corresponding labels

# Load your data here
# early_layer_activations = ...
# late_layer_activations = ...
# gabor_orientations = ...
# class_labels = ...

# Step 3: Dimensionality Reduction (if necessary)
# pca = PCA(n_components=50)
# early_layer_reduced = pca.fit_transform(early_layer_activations)
# late_layer_reduced = pca.fit_transform(late_layer_activations)

# Step 4: Model Training
# Decoding Gabor orientations from early layer activations
# svm_gabor = SVC(kernel='linear')
# svm_gabor.fit(early_layer_reduced, gabor_orientations)

# Decoding class information from late layer activations
# svm_class = SVC(kernel='linear')
# svm_class.fit(late_layer_reduced, class_labels)

# Step 5: Evaluation
# Predicting and evaluating on the same data (ideally, use separate train and test sets)
# gabor_predictions = svm_gabor.predict(early_layer_reduced)
# class_predictions = svm_class.predict(late_layer_reduced)

# gabor_accuracy = accuracy_score(gabor_orientations, gabor_predictions)
# class_accuracy = accuracy_score(class_labels, class_predictions)

# print(f"Gabor Orientation Decoding Accuracy: {gabor_accuracy * 100:.2f}%")
# print(f"Class Information Decoding Accuracy: {class_accuracy * 100:.2f}%")
