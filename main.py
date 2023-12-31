import tkinter as tk
from classification import Classification
from clustering import Clustering

class Main:
    def __init__(self, master):
        self.master = master
        master.title("Data Mining")
        master.geometry("900x550")
        
        self.label = tk.Label(master, text="Machine learning", font=('helvetica', 10,'bold'))
        self.label.place(relx=0.5, rely=0.05, anchor="center")
        
        self.clustering_button = tk.Button(master, text="Clustering", command=self.open_clustering,bg='pale turquoise', fg='black', font=('helvetica', 10, 'bold'), width=10)
        self.clustering_button.place(x=335 - 20, y=120 - 20)
        
        self.classification_button = tk.Button(master, text="Classification", command=self.open_classification,bg='green yellow', fg='black', font=('helvetica', 10, 'bold'), width=10)
        self.classification_button.place(x=500 - 20, y=120 - 20)
        
    
    def open_clustering(self):
        root = tk.Tk()
        interface = Clustering(root)
        root.mainloop()

    def open_classification(self):
        root = tk.Tk()
        interface = Classification(root)
        root.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    interface_selector = Main(root)
    root.mainloop()
