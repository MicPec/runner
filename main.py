import streamlit as st
import pandas as pd
import json
import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv
from langfuse.decorators import observe
from langfuse.openai import OpenAI
import boto3
from pycaret.regression import load_model, predict_model
import plotly.graph_objects as go

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

# Initialize S3 client
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3"),
)
BUCKET_NAME = "runner"


@dataclass
class RunnerInfo:
    age: int
    sex: str
    time_5k: float  # time in seconds
    birth_year: int = None
    age_category: str = None

    def __post_init__(self):
        # Calculate birth year if not provided
        if self.birth_year is None:
            self.birth_year = datetime.now().year - self.age
        # Calculate age category, actually i'm proud of this XD
        self.age_category = self.sex + str(max(min((datetime.now().year - self.birth_year) // 10 * 10, 80), 20))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Płeć": self.sex.upper(),  # Match the model's expected format
            "Rocznik": self.birth_year,
            "5 km Czas": self.time_5k,
            "Kategoria wiekowa": self.age_category,
        }

    @staticmethod
    def parse_time(time_str: str) -> float:
        """Parse time string in format MM:SS to seconds"""
        if isinstance(time_str, (int, float)):
            return float(time_str)

        # Check if time is in MM:SS format
        match = re.match(r"^(\d+):(\d{2})$", time_str)
        if match:
            minutes, seconds = map(int, match.groups())
            return minutes * 60 + seconds

        # Try to convert directly to float (assuming seconds)
        try:
            return float(time_str)
        except ValueError:
            raise ValueError(f"Could not parse time: {time_str}")

    @staticmethod
    def format_time(seconds: float) -> str:
        """Format seconds to MM:SS"""
        minutes = int(seconds // 60)
        remaining_seconds = int(seconds % 60)
        return f"{minutes}:{remaining_seconds:02d}"


def time_to_seconds(time: str) -> int:
    if pd.isnull(time) or time in ["DNS", "DNF"]:
        return None
    time = time.split(":")
    return int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])


def seconds_to_time(seconds: int) -> str:
    if pd.isnull(seconds):
        return None
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def predict_finish_time(df: pd.DataFrame, age_ctg: str, sex: str, time_5km: int):
    df_pred = df[["Kategoria wiekowa", "Płeć", "5 km Czas", "Czas"]].copy()
    df_pred["Czas_seconds"] = df_pred["Czas"].apply(time_to_seconds)
    df_pred["5 km Czas_seconds"] = df_pred["5 km Czas"].apply(time_to_seconds)

    df_filtered = df_pred[(df_pred["Kategoria wiekowa"] == age_ctg) & (df_pred["Płeć"] == sex)]
    if len(df_filtered) == 0:
        df_filtered = df_pred
    df_filtered = df_filtered.dropna(subset=["5 km Czas_seconds", "Czas_seconds"])

    # check for exact match
    exact_match = df_filtered[df_filtered["5 km Czas_seconds"] == time_5km]
    if len(exact_match) > 0:  # count the mean if more than one
        mean_time_seconds = exact_match["Czas_seconds"].mean()
        return mean_time_seconds

    # find closest one
    df_filtered["time_diff"] = abs(df_filtered["5 km Czas_seconds"] - time_5km)
    closest = df_filtered.loc[df_filtered["time_diff"].idxmin()]

    return closest["Czas_seconds"]


@observe()
def parse_user_input(text: str) -> RunnerInfo:
    """Parse user input using OpenAI to extract runner information"""
    try:
        time_pattern = re.compile(r"(\d+):(\d{2})")
        time_match = time_pattern.search(text)
        time_hint = ""

        if time_match:
            minutes, seconds = map(int, time_match.groups())
            time_seconds = minutes * 60 + seconds
            time_hint = f"\n\nNote: The time '{time_match.group()}' in the text is equivalent to {time_seconds} seconds.\n\nCurrent year is {datetime.now().year}"

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"Extract information about the runner from the text. Return a JSON object with the following fields: age (int - age), sex (string 'M' for male or 'K' for female), time_5k (string in MM:SS format or number of seconds as float), birth_year (int - year of birth, optional).{time_hint}",
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )

        data = json.loads(response.choices[0].message.content)

        if data.get("time_5k") is None:
            raise ValueError("Nie udało się znaleźć czasu 5 km w tekście.")
        # Parse time if it's in MM:SS format
        original_time = data.get("time_5k")
        if isinstance(original_time, str):
            data["time_5k"] = RunnerInfo.parse_time(original_time)

        if data.get("time_5k") < 750 or data.get("time_5k") > 7200:
            raise ValueError("Nieprawidłowy czas 5 km.")
        if data.get("sex") is None or data.get("sex") not in ["M", "K"]:
            raise ValueError("Nieprawidłowa płec.")
        if data.get("age") is None or data.get("age") < 18 or data.get("age") > 105:
            raise ValueError("Nieprawidłowy wiek.")

        if "birth_year" in data:
            runner = RunnerInfo(
                age=int(data.get("age")),
                sex=data.get("sex"),
                time_5k=float(data.get("time_5k")),
                birth_year=int(data.get("birth_year")),
            )
        else:
            runner = RunnerInfo(age=int(data.get("age")), sex=data.get("sex"), time_5k=float(data.get("time_5k")))

        return runner
    except ValueError as e:
        st.error(f"Zły format danych: {str(e)}")
        raise
    except Exception as e:
        st.error(f"Wystąpił błąd: {str(e)}")
        raise


@st.cache_data
def get_model():
    model_path = "runner_model"
    try:
        model = load_model(model_path, platform="aws", authentication={"bucket": BUCKET_NAME})
        st.success("Załadowano istniejący model")
        return model
    except Exception as e:
        st.error(f"Błąd ładowania modelu: {str(e)}.")
        raise Exception("Nie znaleziono modelu.")


def predict_half_marathon_time(runner_info: RunnerInfo, model, df_hw=None):
    input_data = pd.DataFrame([runner_info.to_dict()])

    age_category = runner_info.sex + str(max(min((datetime.now().year - runner_info.birth_year) // 10 * 10, 80), 20))
    input_data["Kategoria wiekowa"] = input_data.apply(
        lambda row: row["Płeć"] + str(max(min((datetime.now().year - row["Rocznik"]) // 10 * 10, 80), 20)), axis=1
    )

    # ML model prediction
    pred = predict_model(model, data=input_data)
    pred_sec_ml = pred["prediction_label"].iloc[0]

    # mean algorithm prediction
    if df_hw is not None:
        time_5km_seconds = runner_info.time_5k
        pred_sec_mean = predict_finish_time(df_hw, age_category, runner_info.sex, time_5km_seconds)
        return (
            pred_sec_ml,  # ML seconds
            pred_sec_mean,  # Mean seconds
        )

    return pred_sec_ml, None


def main():
    if "runner_info" not in st.session_state:
        st.session_state.runner_info = None
    if "page" not in st.session_state:
        st.session_state.page = "input"
    if "prediction_results" not in st.session_state:
        st.session_state.prediction_results = None

    st.set_page_config(
        page_title="Kalkulator Czasu Półmaratonu",
        page_icon="🏃",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Custom CSS for better styling
    def local_css():
        st.markdown(
            """
        <style>
            .main-header {
                font-size: 2.5rem !important;
                text-align: center;
                margin-bottom: 1rem;
                font-weight: 700;
            }
            .sub-header {
                font-size: 1.8rem !important;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
                font-weight: 600;
            }
            .section-header {
                font-size: 1.4rem !important;
                margin-top: 1rem;
                margin-bottom: 0.5rem;
                font-weight: 500;
            }
            .highlight-box {
                padding: 1.5rem;
                border-radius: 0.5rem;
                border-left: 0.5rem solid;
                margin: 1rem 0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            .success-box {
                padding: 1.5rem;
                border-radius: 0.5rem;
                border-left: 0.5rem solid;
                margin: 1rem 0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            .info-box {
                padding: 1.5rem;
                border-radius: 0.5rem;
                border-left: 0.5rem solid;
                margin: 1rem 0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            .warning-box {
                padding: 1.5rem;
                border-radius: 0.5rem;
                border-left: 0.5rem solid;
                margin: 1rem 0;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
            }
            .metric-container {
                text-align: center;
                padding: 1rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                transition: all 0.3s ease;
            }
            .metric-container:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            }
            .metric-label {
                font-size: 0.9rem;
                margin-bottom: 0.3rem;
            }
            .metric-value {
                font-size: 1.4rem;
                font-weight: 600;
            }
            .divider {
                height: 1px;
                margin: 2rem 0;
            }
            /* Tabs styling */
            .stTabs [data-baseweb="tab-list"] {
                gap: 2rem;
            }
            .stTabs [data-baseweb="tab"] {
                height: 3rem;
                white-space: pre-wrap;
                border-radius: 0.5rem 0.5rem 0 0;
                padding: 0.5rem 1rem;
                font-weight: 500;
            }
            /* Button styling */
            .stButton > button {
                border-radius: 0.5rem;
                padding: 0.5rem 1rem;
                font-weight: 600;
                border: none;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                transition: all 0.3s;
            }
            .stButton > button:hover {
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
                transform: translateY(-2px);
            }
            /* Input styling */
            .stTextInput > div > div > input,
            .stNumberInput > div > div > input,
            .stTextArea > div > div > textarea {
                border-radius: 0.5rem;
                border: 2px solid;
                padding: 0.5rem 1rem;
                transition: all 0.3s;
            }
            .stTextInput > div > div > input:focus,
            .stNumberInput > div > div > input:focus,
            .stTextArea > div > div > textarea:focus {
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }
            .stSelectbox > div > div {
                border-radius: 0.5rem;
                border: 2px solid;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )

    local_css()

    if st.session_state.page == "input":
        display_input_page()
    elif st.session_state.page == "results":
        display_results_page()


@st.cache_data
def load_dataframes():
    try:
        df_hw23 = pd.read_csv(f"s3://{BUCKET_NAME}/halfmarathon_wroclaw_2023__final.csv", sep=";")
        df_hw24 = pd.read_csv(f"s3://{BUCKET_NAME}/halfmarathon_wroclaw_2024__final.csv", sep=";")
        df_hw = pd.concat([df_hw23, df_hw24], ignore_index=True)
        return df_hw
    except Exception as e:
        st.markdown(
            f'<div class="warning-box">⚠️ Nie udało się załadować danych półmaratonu: {str(e)}</div>',
            unsafe_allow_html=True,
        )
        return None


def display_input_page():
    st.markdown('<h1 class="main-header">🏃 Kalkulator Czasu Półmaratonu</h1>', unsafe_allow_html=True)

    st.markdown(
        """
    <div class="info-box">
        <h3 style="margin-top: 0;">⏱️ Oszacuj swój czas półmaratonu</h3>
        <p>Ten kalkulator wykorzystuje dwa podejścia do przewidywania czasu półmaratonu:</p>
        <ul>
            <li><b>🤖 Model uczenia maszynowego</b> - trenowany na danych z półmaratonów Wrocławskich</li>
            <li><b>📊 Algorytm średniej</b> - bazujący na średnich czasach biegaczy o podobnym profilu</li>
        </ul>
        <p>Podaj swoje dane, aby otrzymać spersonalizowaną prognozę!</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    df_hw = load_dataframes()
    data_loaded = df_hw is not None

    tabs = st.tabs(["📋 Formularz", "💬 Język naturalny"])

    with tabs[0]:  # Form input
        st.markdown('<h3 class="sub-header">📝 Wprowadź swoje dane</h3>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<p style="font-weight: 500; margin-bottom: 0.5rem;">👤 Wiek</p>', unsafe_allow_html=True)
            age = st.number_input("Wiek", min_value=18, max_value=105, value=30, label_visibility="collapsed")

            st.markdown(
                '<p style="font-weight: 500; margin-bottom: 0.5rem; margin-top: 1rem;">⚧️ Płeć</p>',
                unsafe_allow_html=True,
            )
            sex = st.selectbox("Płeć", options=["M", "K"], label_visibility="collapsed")

        with col2:
            st.markdown(
                '<p style="font-weight: 500; margin-bottom: 0.5rem;">⏱️ Czas 5 km (MM:SS)</p>', unsafe_allow_html=True
            )
            time_5k_str = st.text_input("Czas 5 km (MM:SS)", value="25:00", label_visibility="collapsed")
            try:
                time_5k = RunnerInfo.parse_time(time_5k_str)  # Already returns seconds
            except ValueError:
                st.markdown(
                    '<div class="warning-box">❌ Nieprawidłowy format czasu. Proszę użyć formatu MM:SS.</div>',
                    unsafe_allow_html=True,
                )
                time_5k = None

        if time_5k is not None:
            model_sex = "M" if sex == "M" else "K"

        if st.button("Akceptuj dane", key="submit_btn", use_container_width=True):
            st.session_state.runner_info = RunnerInfo(age=age, sex=model_sex, time_5k=time_5k)

    with tabs[1]:  # NLP input
        st.markdown('<h3 class="sub-header">💬 Opisz swoje dane</h3>', unsafe_allow_html=True)

        st.markdown(
            """
        <div class="highlight-box">
            <p>Opisz siebie i swój czas na 5 km używając naturalnego języka. Na przykład:</p>
            <ul>
                <li>"Jestem 35-letnim mężczyzną, a mój czas na 5 km to 22:30."</li>
                <li>"Mam 28 lat, jestem kobietą i biegam 5 km w 27 minut."</li>
            </ul>
        </div>
        """,
            unsafe_allow_html=True,
        )

        user_input = st.text_area(
            "Opisz siebie i swój czas na 5 km",
            "Jestem 35-letnim mężczyzną, a mój czas na 5 km to 22:30.",
            height=120,
            label_visibility="collapsed",
        )

        analyze_button = st.button("🔍 Analizuj dane", key="analyze_btn", use_container_width=True)

        if analyze_button:
            with st.spinner("Analizowanie danych..."):
                try:
                    st.session_state.runner_info = parse_user_input(user_input)
                    formatted_time = RunnerInfo.format_time(st.session_state.runner_info.time_5k)
                    st.markdown(
                        f"""
                        <div class="success-box">
                            <h3 style="margin-top: 0;">✅ Przeanalizowano pomyślnie</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 0.5rem; font-weight: bold;">👤 Wiek:</td>
                                    <td style="padding: 0.5rem;">{st.session_state.runner_info.age} lat</td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem; font-weight: bold;">⚧️ Płeć:</td>
                                    <td style="padding: 0.5rem;">{st.session_state.runner_info.sex}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem; font-weight: bold;">⏱️ Czas 5 km:</td>
                                    <td style="padding: 0.5rem;">{formatted_time}</td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem; font-weight: bold;">📅 Rok urodzenia:</td>
                                    <td style="padding: 0.5rem;">{st.session_state.runner_info.birth_year}</td>
                                </tr>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                except Exception as e:
                    st.markdown(
                        f'<div class="warning-box">❌ Nie udało się przeanalizować danych: {str(e)}</div>',
                        unsafe_allow_html=True,
                    )

    if st.session_state.runner_info:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            <div style="text-align: center;">
                <h3 style="margin-bottom: 1rem;">Gotowy do obliczenia czasu?</h3>
            </div>
            """,
                unsafe_allow_html=True,
            )
        with col2:
            estimate_button = st.button("🚀 Oszacuj czas półmaratonu", key="estimate_btn", use_container_width=True)

        if estimate_button:
            with st.spinner("Obliczanie czasu..."):
                try:
                    model = get_model()

                    time_sec_ml, time_sec_mean = predict_half_marathon_time(
                        st.session_state.runner_info, model, df_hw if data_loaded else None
                    )

                    st.session_state.prediction_results = {
                        "ml": {"minutes": time_sec_ml / 60, "formatted": seconds_to_time(time_sec_ml)},
                        "mean": {"minutes": time_sec_mean / 60, "formatted": seconds_to_time(time_sec_mean)}
                        if time_sec_mean is not None
                        else None,
                    }

                    st.session_state.page = "results"
                    st.rerun()

                except Exception as e:
                    st.error(f"Nie udało się oszacować czasu półmaratonu: {str(e)}")


def display_results_page():
    st.markdown('<h1 class="main-header">🏆 Wyniki Predykcji Czasu Półmaratonu</h1>', unsafe_allow_html=True)

    if st.session_state.prediction_results is None:
        st.markdown(
            '<div class="warning-box">❌ Brak wyników predykcji. Wróć do strony głównej.</div>', unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("← Wróć do strony głównej", use_container_width=True):
                st.session_state.page = "input"
                st.rerun()
        return

    results = st.session_state.prediction_results

    runner = st.session_state.runner_info
    formatted_time = RunnerInfo.format_time(runner.time_5k)

    st.markdown('<h2 class="sub-header">👤 Dane biegacza</h2>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-label">👤 Wiek</div>
            <div class="metric-value">{runner.age} lat</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-label">⚧️ Płeć</div>
            <div class="metric-value">{runner.sex}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-label">⏱️ Czas 5 km</div>
            <div class="metric-value">{formatted_time}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
        <div class="metric-container">
            <div class="metric-label">📊 Kategoria</div>
            <div class="metric-value">{runner.age_category}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.markdown('<h2 class="sub-header">🔮 Wyniki predykcji</h2>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ML model results
    with col1:
        st.markdown('<h3 class="section-header">🤖 Modelem ML</h3>', unsafe_allow_html=True)
        st.markdown(
            f"""
        <div style="text-align: center; margin: 1rem 0;">
            <div style="font-size: 2.5rem; font-weight: 700;">{results['ml']['formatted']}</div>
            <div style="font-size: 1.2rem;">({results['ml']['minutes']:.2f} minut)</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Mean algorithm results if available
    with col2:
        if results["mean"] is not None:
            st.markdown('<h3 class="section-header">📊 Modelem średniej</h3>', unsafe_allow_html=True)
            st.markdown(
                f"""
            <div style="text-align: center; margin: 1rem 0;">
                <div style="font-size: 2.5rem; font-weight: 700;">{results['mean']['formatted']}</div>
                <div style="font-size: 1.2rem;">({results['mean']['minutes']:.2f} minut)</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    if results["mean"] is not None:
        st.markdown('<h3 class="section-header">📈 Porównanie predykcji</h3>', unsafe_allow_html=True)

        # Create data for the chart
        methods = ["Model ML", "Algorytm średniej"]
        times = [results["ml"]["minutes"], results["mean"]["minutes"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=methods,
                y=times,
                text=[f"{t:.2f} min" for t in times],
                textposition="auto",
                marker_color=["#3498db", "#2ecc71"],
            )
        )
        fig.update_layout(
            title="Porównanie czasów predykcji (w minutach)",
            xaxis_title="Metoda predykcji",
            yaxis_title="Czas (minuty)",
            height=400,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        diff = abs(results["ml"]["minutes"] - results["mean"]["minutes"])
        diff_percent = (diff / results["ml"]["minutes"]) * 100

        st.markdown(
            f"""
        <div class="info-box">
            <h3 style="margin-top: 0;">📏 Różnica między predykcjami</h3>
            <p><b>{diff:.2f} minut</b> ({diff_percent:.1f}%)</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<h3 class="section-header">⏱️ Tempo biegu</h3>', unsafe_allow_html=True)

    ml_pace_min_km = results["ml"]["minutes"] / 21.0975
    ml_pace_min = int(ml_pace_min_km)
    ml_pace_sec = int((ml_pace_min_km - ml_pace_min) * 60)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=ml_pace_min_km,
            title={"text": "Tempo (min/km)"},
            delta={"reference": 5.5},  # Reference pace for comparison
            gauge={
                "axis": {"range": [3, 8], "tickwidth": 1},
                "bar": {"color": "#3498db"},
                "steps": [
                    {"range": [3, 4], "color": "#1abc9c"},  # Fast
                    {"range": [4, 5.5], "color": "#2ecc71"},  # Good
                    {"range": [5.5, 6.5], "color": "#f1c40f"},  # Average
                    {"range": [6.5, 8], "color": "#e74c3c"},  # Slow
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": ml_pace_min_km},
            },
        )
    )
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"""
    <div class="success-box">
        <h3 style="margin-top: 0;">🏃‍♂️ Twoje przewidywane tempo</h3>
        <p style="font-size: 1.5rem; font-weight: 700;">{ml_pace_min}:{ml_pace_sec:02d} min/km</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">📋 Rekomendacje treningowe</h2>', unsafe_allow_html=True)

    if results["ml"]["minutes"] < 90:
        recommendation = """
        <div class="highlight-box">
            <h3 class="section-header">🔥 Świetny wynik!</h3>
            <p>Jesteś na dobrej drodze do znakomitego czasu w półmaratonie! Twoje tempo plasuje Cię w czołówce biegaczy.</p>
            <h4>Zalecenia:</h4>
            <ul>
                <li>🏋️‍♂️ Skup się na treningu szybkościowym i wytrzymałościowym</li>
                <li>⚡ Wprowadź interwały o wysokiej intensywności</li>
                <li>🧘‍♂️ Pamiętaj o odpowiedniej regeneracji między intensywnymi treningami</li>
                <li>👨‍🏫 Rozważ konsultację z trenerem biegowym, aby dopracować technikę</li>
            </ul>
        </div>
        """
    elif results["ml"]["minutes"] < 120:
        recommendation = """
        <div class="highlight-box">
            <h3 class="section-header">👍 Dobre tempo!</h3>
            <p>Twój przewidywany czas jest solidny! Z odpowiednim treningiem możesz go jeszcze poprawić.</p>
            <h4>Zalecenia:</h4>
            <ul>
                <li>🏃‍♂️ Rozważ włączenie do treningu więcej biegów tempowych</li>
                <li>🛣️ Dodaj długie wybiegania (16-18 km) raz w tygodniu</li>
                <li>💪 Pracuj nad siłą mięśni nóg i stabilizacją korpusu</li>
                <li>👟 Zwróć uwagę na technikę biegu i ekonomię ruchu</li>
            </ul>
        </div>
        """
    else:
        recommendation = """
        <div class="highlight-box">
            <h3 class="section-header">🚀 Tak trzymaj!</h3>
            <p>Dobra robota! Kontynuując regularny trening, będziesz systematycznie poprawiać swój czas.</p>
            <h4>Zalecenia:</h4>
            <ul>
                <li>🏃‍♀️ Skup się na budowaniu wytrzymałości poprzez dłuższe biegi</li>
                <li>📈 Stopniowo zwiększaj dystans tygodniowy</li>
                <li>🏋️ Wprowadź trening siłowy 1-2 razy w tygodniu</li>
                <li>⏰ Pamiętaj o regularnych treningach i odpowiednim odpoczynku</li>
            </ul>
        </div>
        """

    st.markdown(recommendation, unsafe_allow_html=True)

    if st.button("🔄 Wróć do strony głównej", use_container_width=True):
        st.session_state.runner_info = None
        st.session_state.prediction_results = None
        st.session_state.page = "input"
        st.rerun()


if __name__ == "__main__":
    main()
