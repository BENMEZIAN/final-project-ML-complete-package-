import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class Regression:
    def __init__(self, master):
        self.master = master
        master.title("Supervised regression")
        master.geometry("900x550")
        
        self.label = tk.Label(master, text="Supervised regression", font=('helvetica', 10,'bold'))
        self.label.place(relx=0.5, rely=0.05, anchor="center")

        # Create buttons
        self.import_button = tk.Button(master, text="Import Dataset", command=self.import_dataset, bg='green', fg='black',font=('helvetica', 10, 'bold'))
        self.import_button.place(x=180, y=50)

        self.split_button = tk.Button(master, text="Split Dataset", command=self.split_dataset,bg='orange red', fg='black', font=('helvetica', 10, 'bold'))
        self.split_button.place(x=330, y=50)

        self.run_regression_button = tk.Button(master, text="Run Logistic Regression",command=self.run_regression,bg='purple', fg='black', font=('helvetica', 10, 'bold'))
        self.run_regression_button.place(x=500, y=50)

        # Variable to store the dataset
        self.dataset = None

    def import_dataset(self):
        # Open a file dialog to choose the dataset
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

        if file_path:
            # Load the dataset
            self.dataset = pd.read_csv(file_path)
            messagebox.showinfo("Success", "Dataset loaded successfully.")

            # Enable the split button
            self.split_button.config(state=tk.NORMAL)

    def split_dataset(self):
        if self.dataset is not None:
            # Extract features and target variable
            X = self.dataset.drop('Class variable', axis=1)
            y = self.dataset['Class variable']

            # Split the data into training and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize the features
            scaler = StandardScaler()
            self.X_train_scaled = scaler.fit_transform(self.X_train)
            self.X_test_scaled = scaler.transform(self.X_test)

            messagebox.showinfo("Success", "Dataset split successfully.")

            # Enable the regression button
            self.run_regression_button.config(state=tk.NORMAL)

    def run_regression(self):
        # Create and train the logistic regression model
        model = LogisticRegression()
        model.fit(self.X_train_scaled, self.y_train)

        # Make predictions on the test set
        y_pred = model.predict(self.X_test_scaled)

        # Evaluate the model
        accuracy = accuracy_score(self.y_test, y_pred)
        messagebox.showinfo("Result", f"Accuracy: {accuracy:.2f}")

        # Display classification report
        report = classification_report(self.y_test, y_pred)
        messagebox.showinfo("Classification Report", report)
