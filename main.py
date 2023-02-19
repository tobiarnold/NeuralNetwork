import streamlit as st
import random
import warnings
import pandas as pd
#from st_aggrid import AgGrid,ColumnsAutoSizeMode
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras

def main():
    st.set_page_config(page_title="Titanic",page_icon="🚢",layout="wide")
    warnings.filterwarnings("ignore")
    random.seed(42)
    hide_streamlit_style = """
             <style>
              div.block-container{padding-top:2rem;}
               div[data-testid="stToolbar"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
                }
             </style>
             """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    st.title("🚢 Überleben auf der Titanic")
    st.subheader("👩‍💻 Business Intelligence Gruppe 7 Artificial Neural Networks und Deep Learning")
    st.info("""
                      - Die nachfolgende Anwendung berechnet mittels eines **Neuronalen Netzes** die theoretische Überlebenswahrscheinlichkeit auf der Titanic.
                      - Das vorliegende Neuronale Netz verfügt über 3 Schichten (Eingabeschicht, verborgene Schicht und Ausgabeschicht).
                      - Um den Vorgang zu starten, gebt bitte die erforderlichen Parameter in der Sidebar auf der linken Seite ein und klickt auf "Eingaben bestätigen".
                       """)
    st.markdown("""----""")
    st.subheader("💾 Datengrundlage")
    try:
        df=pd.read_csv(r"https://raw.githubusercontent.com/tobiarnold/NeuralNetwork/main/titanic_new.csv")
        #AgGrid(df,height=300,columns_auto_size_mode=ColumnsAutoSizeMode.FIT_CONTENTS)
        df_style = df[df["überlebt (0=Nein; 1=Ja)"]==1]
        slice_ = pd.IndexSlice[df_style.index, df_style.columns]
        df_show = df.style.set_properties(**{"background-color": " #ffffa1"}, subset=slice_)
        st.dataframe(df_show)
    except:
        st.write("Tabelle konnte nicht geladen werden, bitte App neu laden.")
    st.markdown("""----""")
    st.subheader("📊 Diagramme")
    try:
        config = {"displayModeBar": False}
        col1, col2, col3 = st.columns(3)
        with col1:
            fig2 = px.box(df, y="Alter",title="Altersverteilung",
                          color_discrete_sequence=px.colors.qualitative.Vivid)
            st.plotly_chart(fig2, use_container_width=True, config=config)
        with col2:
            fig3 = px.histogram(df, x="Geschlecht (0=männlich; 1=weiblich)",
                                color="Geschlecht (0=männlich; 1=weiblich)",
                                color_discrete_sequence=px.colors.qualitative.Vivid, 
                                title="Geschlechterverteilung")
            fig3.update_layout(xaxis=dict(tickmode='array', 
                               tickvals=[0, 1], ticktext=["männlich", "weiblich"]))
            fig3.update_layout(showlegend=False)
            fig3.update_layout(xaxis_title="")
            fig3.update_layout(yaxis_title="Anzahl")
            st.plotly_chart(fig3, use_container_width=True, config=config)
        with col3:
            fig4 = px.histogram(df, x="Passagierklasse (1,2,3)",
                                color="Passagierklasse (1,2,3)", 
                                color_discrete_sequence=px.colors.qualitative.Vivid,
                                title="Verteilung Passagierklassen")
            fig4.update_layout(showlegend=False)
            fig4.update_layout(xaxis_title="")
            fig4.update_layout(yaxis_title="Anzahl")
            st.plotly_chart(fig4, use_container_width=True, config=config)
    except:
        st.write("Diagramme konnten nicht geladen werden, bitte App neu laden.")
    try:
        with st.form(key='Form'):
            with st.sidebar:
                st.sidebar.header("💡 Parameter auswählen:")
                epochs=st.sidebar.slider("Anzahl der Epochen (Durchläufe auswählen):", 1, 100, 30, 1)
                neuro=st.sidebar.slider("Anzahl der Neuronen für Input und Hidden Schicht (Hälfte der Neuronen der Input Schicht) wählen:"
                                        , 2, 256, 32, 2)
                aktiv = st.selectbox("Aktivierungsfunktion für Input und verborgene Schicht wählen:",
                                     options=["relu", "sigmoid", "tanh"], index=0)
                optimizer=st.selectbox("Optimierer wählen:", 
                                       options=["Adam","Adagrad","RMSprop","SGD"], index=0)
                alter = st.sidebar.slider("Alter:", 1, 80, 30, 1)
                geschlecht = st.radio("Geschlecht auswählen:", 
                                      options=["männlich", "weiblich"], index=1)
                klasse = st.sidebar.selectbox("Passagierklasse auswählen:",
                                              options=[1,2,3], index=1)
                submitted = st.form_submit_button(label="Eingaben bestätigen")
        st.markdown("""----""")
        st.subheader("🐱‍💻 Künstliches neuronales Netz")
        if submitted:
            with st.spinner("Bitte warten Neuronales Netz wird ausgeführt"):
                x = df.drop(["Name", "überlebt (0=Nein; 1=Ja)"], axis=1)
                y = df["überlebt (0=Nein; 1=Ja)"]
                x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
                tf.random.set_seed(42)
                model = keras.Sequential([
                    keras.layers.Dense(neuro, activation=aktiv, input_shape=(x_train.shape[1],)),
                    keras.layers.Dense(neuro/2, activation=aktiv),
                    keras.layers.Dense(1, activation="sigmoid")
                ])
                model.compile(optimizer=optimizer,loss="binary_crossentropy",metrics=["accuracy"])
                history=model.fit(x_train, y_train, epochs=epochs,validation_data=(x_test, y_test))
                if geschlecht=="männlich":
                    geschlecht=0
                else:
                    geschlecht=1
                new_sample = [[geschlecht, alter, klasse]]
                prediction = model.predict(new_sample)
                prediction1=prediction*100
                prediction1=str(prediction1)
                prediction1=prediction1[2:7]
                st.success("#### Deine Überlebenswahrscheinlichkeit beträgt: "+ prediction1+ " %")
                model.summary(print_fn=lambda x: st.text(x))
                y_predictions = model.predict(x_test)
                y_predictions = (y_predictions > 0.5)
                st.text("Model Report:\n " + classification_report(y_test, y_predictions))
                confusionmatrix = confusion_matrix(y_test, y_predictions)
                accuracy = accuracy_score(y_test, y_predictions)
                st.write("Test accuracy:", str(accuracy))
                fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
                ax[0].plot(history.history["loss"], label="Training loss")
                ax[0].plot(history.history["val_loss"], label="Validation loss")
                ax[0].set_ylabel("Fehler")
                ax[0].set_xlabel("Epochen")
                ax[0].set_title("Training- und Validation loss")
                ax[0].legend()
                sns.heatmap(confusionmatrix, annot=True, cmap="viridis", fmt=".0f", ax=ax[1], cbar=True)
                ax[1].set_xlabel("Predicted")
                ax[1].set_ylabel("Actual")
                ax[1].set_title("Confusion Matrix")
                fig.tight_layout()
                st.pyplot(fig)
    except:
        st.write("Fehler bei der Erstellung des Neuronalen Netzes, bitte App neu laden.")
                    
if __name__ == '__main__':
      main()
        
