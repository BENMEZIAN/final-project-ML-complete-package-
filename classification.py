import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import tensorflow as tf
from tensorflow import keras
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report


class Classification:
    def __init__(self, master):
        self.master = master
        master.title("Supervised classification")
        master.geometry("900x550")
        
        self.label = tk.Label(master, text="Supervised classification", font=('helvetica', 10,'bold'))
        self.label.place(relx=0.5, rely=0.05, anchor="center")

        self.open_button = tk.Button(master, text="Import CSV file", command=self.open_dataset, bg='green', fg='black',font=('helvetica', 10, 'bold'))
        self.open_button.place(x=180, y=50)

        self.clean_and_normalize_data_button = tk.Button(master, text="Preprocess dataset", command=self.clean_and_normalize_data,bg='blue', fg='black', font=('helvetica', 10, 'bold'))
        self.clean_and_normalize_data_button.place(x=330, y=50)

        self.display_button = tk.Button(master, text="Display dataset", command=self.display_normalized_data,bg='orange red', fg='black', font=('helvetica', 10, 'bold'))
        self.display_button.place(x=500, y=50)
        
        self.knn_button = tk.Button(master, text="Apply KNN", command=self.run_knn_classification,bg='purple', fg='black', font=('helvetica', 10, 'bold'))
        self.knn_button.place(x=180, y=120)

        self.naive_bayes_button = tk.Button(master, text="Apply Naive Bayes",command=self.run_naive_bayes_classification, bg='orange', fg='black',font=('helvetica', 10, 'bold'))
        self.naive_bayes_button.place(x=300, y=120)
        
        self.decision_tree_button = tk.Button(master, text="Apply Decision Tree",command=self.run_decisionTree_classification, bg='magenta', fg='black',font=('helvetica', 10, 'bold'))
        self.decision_tree_button.place(x=500, y=120)
        
        self.neural_network_button = tk.Button(master, text="Apply Neural Network",command=self.run_neuralNetwork, bg='yellow', fg='black',font=('helvetica', 10, 'bold'))
        self.neural_network_button.place(x=180, y=200)
        
        self.svm_button = tk.Button(master, text="Apply Support Vector Machine",command=self.run_svm_classification, bg='dark sea green', fg='black',font=('helvetica', 10, 'bold'))
        self.svm_button.place(x=400, y=200)
        
        self.data = None
        self.df_normalized = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def open_dataset(self):
        filename = filedialog.askopenfilename()
        if filename:
            self.data = pd.read_csv(filename)
            messagebox.showinfo("Info", "Dataset has been uploaded successfully")
            self.clean_and_normalize_data_button.config(state="normal")

    def split_dataset(self):
        if self.data is not None:
            X = self.data.drop(columns='class')
            y = self.data['class']
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        else:
            messagebox.showwarning("Warning", "Please import a dataset first")

    def clean_and_normalize_data(self):
        # Convert categorical attributes into numerical
        cat_columns = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                cat_columns.append(col)
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col])

        # Normalization
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.data)

        # Display normalized dataset as Pandas DataFrame
        self.df_normalized = pd.DataFrame(data_scaled, columns=self.data.columns)
        messagebox.showinfo("Info", "Dataset has been preprocessed successfully")
        self.display_button.config(state="normal")
        
    def display_normalized_data(self):
        if self.df_normalized is not None:
            top = tk.Toplevel()
            top.title("Normalized Data")
            top.geometry("900x550")
            text = ScrolledText(top)
            text.pack(expand=True, fill='both')
            text.insert('1.0', self.df_normalized.to_string())

    def run_knn_classification(self):
        if self.df_normalized is not None:
            if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
                self.split_dataset()

            # Applying KNN algorithm with euclidean distance
            classifier = KNeighborsClassifier(n_neighbors=27, metric='euclidean')
            classifier.fit(self.X_train, self.y_train)

            # Predict the output for the test set
            y_pred = classifier.predict(self.X_test)

            # Evaluate the model (confusion matrix, accuracy score, f1_score)
            cm = confusion_matrix(self.y_test, y_pred)
            ac = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="weighted")

            # Display the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2'],yticklabels=['Class 1', 'Class 2'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            # Create a bar chart to display the accuracy score
            categories = ['Accuracy']
            values = [ac]

            plt.bar(categories, values, color='blue')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            plt.show()

            messagebox.showinfo("Accuracy", f"Accuracy Score: {ac}")
            messagebox.showinfo("F1 Score", f"F1 Score: {f1}")
        else:
            messagebox.showwarning("Warning", "Please import and preprocess a dataset first")

    def run_naive_bayes_classification(self):
        if self.df_normalized is not None:
            if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
                self.split_dataset()

            # Applying Gaussian Naive Bayes algorithm
            classifier = GaussianNB()
            classifier.fit(self.X_train, self.y_train)

            y_pred = classifier.predict(self.X_test)

            # Evaluate the model (confusion matrix, accuracy score, f1_score)
            cm = confusion_matrix(self.y_test, y_pred)
            ac = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="weighted")

            # Display the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2'],yticklabels=['Class 1', 'Class 2'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            # Create a bar chart to display the accuracy score
            categories = ['Accuracy']
            values = [ac]

            plt.bar(categories, values, color='blue')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            plt.show()

            messagebox.showinfo("Accuracy", f"Accuracy Score: {ac}")
            messagebox.showinfo("F1 Score", f"F1 Score: {f1}")
    
    def run_decisionTree_classification(self):
        if self.df_normalized is not None: 
            if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
                self.split_dataset()
            
            # Assuming 'class' is your target column name
            X = self.df_normalized.drop(columns='class')
            y = self.data['class'] 
            
            # Applying Decision Tree algorithm
            classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 100,max_depth = 4, min_samples_leaf = 5)
            classifier.fit(self.X_train, self.y_train)
            
            # Hyperparameter tuning with GridSearchCV
            param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 4, 5, 6], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
            grid_search = GridSearchCV(DecisionTreeClassifier(random_state=100), param_grid, cv=5)
            grid_search.fit(self.X_train, self.y_train)
            classifier = grid_search.best_estimator_
            
            # Visualize the Decision Tree
            plt.figure(figsize=(12, 6))
            plot_tree(classifier, filled=True, feature_names=X.columns, class_names=list(map(str, classifier.classes_)))
            plt.show()

            # Predict the output for the test set
            y_pred = classifier.predict(self.X_test)

            # Evaluate the model (confusion matrix, accuracy score, f1_score)
            cm = confusion_matrix(self.y_test, y_pred)
            ac = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(y_pred, self.y_test, average="weighted")

            # Display the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2'], yticklabels=['Class 1', 'Class 2'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            messagebox.showinfo("Accuracy", f"Accuracy Score: {ac}")
            messagebox.showinfo("F1 Score", f"F1 Score: {f1}")

    def run_neuralNetwork(self):
        if self.df_normalized is not None:
            if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
                self.split_dataset()
            
            # Set a random seed for reproducibility to ensures that the random initialization of weights 
            # is the same every time you run the code.
            tf.random.set_seed(42)
            np.random.seed(42)
            
            # Define the neural network architecture
            model = keras.Sequential([
                keras.layers.Dense(128, activation='relu', input_dim=self.X_train.shape[1]),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1, activation='sigmoid')  # Adjust for your classification task
            ])

            # Compile the model
            model.compile(optimizer='adam', 
                          loss='binary_crossentropy', 
                          metrics=['accuracy'])

            # Train the model
            model.fit(self.X_train, self.y_train, epochs=5, batch_size=32, validation_split=0.2)
            
            # Evaluate the model on the test set
            y_pred = (model.predict(self.X_test) > 0.5).astype("int32")  # Convert probabilities to binary predictions
            test_loss, test_acc = model.evaluate(self.X_test, self.y_test)

            # Calculate F1 score
            f1 = f1_score(self.y_test, y_pred)
            
            # Calculate confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)

            # Display the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
            
            # Evaluate the model on the test set
            test_loss, test_acc = model.evaluate(self.X_test, self.y_test)
            messagebox.showinfo("Neural Network Accuracy", f"Accuracy Score: {test_acc}")
            messagebox.showinfo("Neural Network f1 score", f"F1 score: {f1}")
        
    def run_svm_classification(self):
        if self.df_normalized is not None:
            if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
                self.split_dataset()

            # Applying SVM algorithm
            classifier = SVC(kernel='linear', C=1.0)
            classifier.fit(self.X_train, self.y_train)

            # Predict the output for the test set
            y_pred = classifier.predict(self.X_test)

            # Evaluate the model (confusion matrix, accuracy score, f1_score)
            cm = confusion_matrix(self.y_test, y_pred)
            ac = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average="weighted")
            report = classification_report(self.y_test, y_pred)

            # Display the confusion matrix as a heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 1', 'Class 2'],yticklabels=['Class 1', 'Class 2'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            # Create a bar chart to display the accuracy score
            categories = ['Accuracy']
            values = [ac]

            plt.bar(categories, values, color='blue')
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.0)
            plt.show()
            
            # Display the classification report
            messagebox.showinfo("Classification Report", f"Classification Report:\n\n{report}")
            messagebox.showinfo("Accuracy", f"Accuracy Score: {ac}")
            messagebox.showinfo("F1 Score", f"F1 Score: {f1}")
