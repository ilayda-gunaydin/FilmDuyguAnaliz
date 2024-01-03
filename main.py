import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVR
import tkinter as tk

def tahmin_yap():
    film_ismi = film_isim_giris.get()
    yorum = yorum_giris.get()
    yorum_vector = vectorizer.transform([yorum])
    duygu_puani = svr.predict(yorum_vector)[0]
    sonuc_metni.config(text=f"{film_ismi} filmi için Yorumunuzun Duygu Puanı: {duygu_puani}")

    if duygu_puani < 2.5:
        emoji.config(text="OLUMSUZ YORUM 🙁", font=("Arial", 24))
    elif duygu_puani >= 2.5:
        emoji.config(text="OLUMLU YORUM 😊", font=("Arial", 24))

veri_seti = pd.read_csv('/Users/ilayda/PycharmProjects/filmYorum/csv/turkish_movie_sentiment_dataset.csv')

veri_seti['point'] = veri_seti['point'].str.replace(',', '.').astype(float)

X = veri_seti['comment']
y = veri_seti['point']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2))
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


svr = LinearSVR()
svr.fit(X_train_vectors, y_train)


root = tk.Tk()
root.title("Film Yorumları Duygu Puanı Tahmini")
root.geometry("900x900")
root.configure(bg="white")


img = tk.PhotoImage(file="/Users/ilayda/Desktop/sinema.png")
gorsel = tk.Label(root, image=img, bg="white")
gorsel.pack()

bosluk = tk.Label(root, text="", bg="white")
bosluk.pack()

ust_cerceve = tk.Frame(root, bg="white")
ust_cerceve.pack()

film_isim_etiket = tk.Label(ust_cerceve, text="Film İsmi:", bg="white", font=("Arial", 24))
film_isim_etiket.grid(row=0, column=0)

film_isim_giris = tk.Entry(ust_cerceve, font=("Arial", 24))
film_isim_giris.grid(row=0, column=1)

yorum_etiket = tk.Label(ust_cerceve, text="Yorumunuz:", bg="white", font=("Arial", 24))
yorum_etiket.grid(row=1, column=0)

yorum_giris = tk.Entry(ust_cerceve, font=("Arial", 24))
yorum_giris.grid(row=1, column=1)

tahmin_buton = tk.Button(root, text="TAHMİN YAP", command=tahmin_yap, font=("Arial", 24, "bold"), bg="green", fg="black")
tahmin_buton.pack()

sonuc_metni = tk.Label(root, text="", bg="white", font=("Arial", 24))
sonuc_metni.pack()

emoji = tk.Label(root, text="", bg="white")
emoji.pack()

root.mainloop()