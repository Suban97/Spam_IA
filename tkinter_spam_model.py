import tkinter as tk
import joblib

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

spam_model.load()
vectorizer.load()
root = tk.Tk()

root.title("yoow")

root.geometry("300x400")





root.mainloop()