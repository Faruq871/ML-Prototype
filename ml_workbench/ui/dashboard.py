from __future__ import annotations
from io import BytesIO
import hashlib
import pandas as pd
import plotly.express as px
import streamlit as st
from ml_workbench.algorithms.catalog import get_algorithms, get_algorithm
from ml_workbench.artifacts.manager import ArtifactManager
from ml_workbench.config import MAX_UPLOAD_MB, RANDOM_STATE
from ml_workbench.data.loader import load_dataframe, validate_dataframe
from ml_workbench.data.profiler import profile_dataframe, missing_summary
from ml_workbench.exceptions import WorkbenchError
from ml_workbench.recommendations.engine import RecommendationEngine
from ml_workbench.training.trainer import train_model
from ml_workbench.training.comparison import compare_models

TASK_LABELS = {
    "classification": "Classification",
    "regression": "Regression",
    "clustering": "Clustering",
    "dimensionality_reduction": "Dimensionality Reduction",
    "anomaly_detection": "Anomaly Detection",
}


def _reset_results():
    for key in ["training_result", "model_download", "report_download", "comparison_result"]:
        st.session_state.pop(key, None)


def _clear_dataset_state():
    # Streamlit callbacks run before the next script execution, so widget/session state can be reset safely here.
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def _render_overview(data: pd.DataFrame):
    profile = profile_dataframe(data)
    cols = st.columns(4)
    cols[0].metric("Rows", f"{profile.rows:,}")
    cols[1].metric("Features", f"{profile.columns:,}")
    cols[2].metric("Missing cells", f"{profile.missing_cells:,}")
    cols[3].metric("Duplicate rows", f"{profile.duplicate_rows:,}")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Data preview")
        st.dataframe(data.head(20), use_container_width=True)
    with right:
        st.markdown("#### Column types")
        types = data.dtypes.astype(str).value_counts().rename_axis("dtype").reset_index(name="columns")
        st.plotly_chart(px.pie(types, names="dtype", values="columns", hole=0.55), use_container_width=True)
    st.markdown("#### Data quality profile")
    st.dataframe(missing_summary(data), use_container_width=True, hide_index=True)
    if profile.constant_columns:
        st.warning(f"Kolom konstan (hanya satu nilai unik): {', '.join(profile.constant_columns)}")
    if profile.high_cardinality_columns:
        st.info("Kolom berkardinalitas tinggi: " + ", ".join(profile.high_cardinality_columns))
    if profile.datetime_columns:
        st.warning("Kolom datetime terdeteksi. Untuk prototype ini, ubah menjadi fitur numerik (mis. year/month/day) sebelum training jika ingin memakainya.")


def _recommendation_panel(data: pd.DataFrame, task: str, target: str | None):
    st.markdown("#### Suggested algorithms")
    recommendations = RecommendationEngine().recommend(data, task, target=target, limit=6)
    for index, rec in enumerate(recommendations, start=1):
        with st.container(border=True):
            top = st.columns([0.09, 0.62, 0.29])
            top[0].markdown(f"**#{index}**")
            top[1].markdown(f"**{rec.name}**")
            top[2].progress(min(rec.score / 100, 1.0), text=f"Heuristic fit: {rec.score:.0f}/100")
            for reason in rec.rationale:
                st.caption("• " + reason)
            for caveat in rec.caveats:
                st.warning(caveat)
    st.caption("Skor adalah prioritas heuristik untuk menentukan kandidat eksperimen, bukan probabilitas model akan menjadi yang terbaik.")


def _show_metrics(metrics: dict):
    scalar_items = [(key, value) for key, value in metrics.items()
                    if isinstance(value, (int, float)) and value is not None]
    if scalar_items:
        cols = st.columns(min(4, len(scalar_items)))
        for index, (key, value) in enumerate(scalar_items):
            rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
            cols[index % len(cols)].metric(key.replace("_", " ").title(), rendered)
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        st.markdown("#### Confusion matrix")
        st.dataframe(pd.DataFrame(cm), use_container_width=False)
    if "explained_variance_ratio" in metrics:
        st.markdown("#### Explained variance ratio")
        st.bar_chart(pd.Series(metrics["explained_variance_ratio"], index=[f"Component {i+1}" for i in range(len(metrics["explained_variance_ratio"]))]))


def run_app():
    st.set_page_config(page_title="ML Workbench", page_icon="🧪", layout="wide")
    st.title("🧪 ML Workbench")
    st.caption("An explainable, human-in-the-loop machine-learning workbench for tabular CSV/XLSX datasets.")

    with st.sidebar:
        st.header("1. Dataset")
        uploaded = st.file_uploader("Upload dataset", type=["csv", "xlsx"], help=f"Maximum file size: {MAX_UPLOAD_MB} MB")
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            signature = f"{uploaded.name}:{hashlib.sha256(file_bytes).hexdigest()}"
            if st.session_state.get("dataset_signature") != signature:
                try:
                    data = load_dataframe(file_bytes, uploaded.name)
                    data = validate_dataframe(data)
                    st.session_state["dataset"] = data
                    st.session_state["dataset_name"] = uploaded.name
                    st.session_state["dataset_signature"] = signature
                    _reset_results()
                except WorkbenchError as exc:
                    st.error(str(exc))
                    st.stop()
        if "dataset" not in st.session_state:
            st.info("Upload file CSV/XLSX untuk memulai. Kamu juga bisa membuat dataset contoh melalui scripts/generate_demo_data.py.")
            st.stop()
        data = st.session_state["dataset"]
        st.success(f"Loaded: {st.session_state.get('dataset_name', 'dataset')}")
        st.caption(f"{data.shape[0]:,} rows × {data.shape[1]:,} columns")
        st.button("Clear dataset", use_container_width=True, on_click=_clear_dataset_state)

    data = st.session_state["dataset"]
    overview_tab, experiment_tab, inference_tab, catalog_tab = st.tabs(["Dataset overview", "Recommend & experiment", "Prediction", "Algorithm catalog"])

    with overview_tab:
        _render_overview(data)

    with experiment_tab:
        st.markdown("### 2. Define the ML problem")
        task = st.selectbox("Learning task", options=list(TASK_LABELS), format_func=lambda x: TASK_LABELS[x], key="task_selection")
        target = None
        if task in {"classification", "regression"}:
            default_target = data.columns[-1]
            target = st.selectbox("Target variable (y)", options=list(data.columns), index=list(data.columns).index(default_target), key="target_selection")
        excluded_default = [target] if target and target in data.columns else []
        excluded = st.multiselect("Additional columns to exclude from features (optional)",
                                  options=[c for c in data.columns if c != target], key="excluded_columns")
        st.caption("Target otomatis dikeluarkan dari fitur. Kolom ID, leakage, atau kolom yang tidak relevan sebaiknya dikecualikan di sini.")
        effective_excluded = list(set(excluded) | set(excluded_default))
        _recommendation_panel(data, task, target)

        task_algorithms = get_algorithms(task)
        algorithm_map = {f"{a.id:02d} · {a.name}": a.id for a in task_algorithms}
        st.markdown("### 3. Select an algorithm")
        default_algorithm = task_algorithms[0].id
        algorithm_label = st.selectbox("Algorithm", options=list(algorithm_map),
                                       index=0, key="algorithm_selection")
        algorithm_id = algorithm_map[algorithm_label]
        algo = get_algorithm(algorithm_id)
        st.caption(f"**Family:** {algo.family}  ·  **Description:** {algo.description}")

        opt1, opt2, opt3 = st.columns(3)
        test_size = opt1.slider("Test set proportion", min_value=0.10, max_value=0.50, value=0.20, step=0.05, disabled=task not in {"classification", "regression"})
        random_state = opt2.number_input("Random state", min_value=0, max_value=999999, value=RANDOM_STATE, step=1)
        n_clusters = opt3.number_input("Number of clusters", min_value=2, max_value=50, value=3, step=1,
                                       disabled=task != "clustering")
        n_components = 2
        if task == "dimensionality_reduction":
            n_components = st.number_input("Number of components", min_value=1, max_value=min(50, max(1, data.shape[1])), value=min(2, max(1, data.shape[1])), step=1)
        if st.button("Run training / experiment", type="primary", use_container_width=True):
            try:
                with st.spinner(f"Running {algo.name}..."):
                    result = train_model(data, algorithm_id, target=target, test_size=test_size,
                                         random_state=int(random_state), n_clusters=int(n_clusters), n_components=int(n_components),
                                         drop_columns=effective_excluded)
                    st.session_state["training_result"] = result
                    manager = ArtifactManager()
                    model_payload = result.estimator
                    model_path = manager.save_model(model_payload, result.algorithm_name, result.task, result.metadata)
                    report_path = manager.save_report({"metrics": result.metrics, "metadata": result.metadata}, result.algorithm_name)
                    st.session_state["model_download"] = model_path.read_bytes()
                    st.session_state["report_download"] = report_path.read_bytes()
                    st.session_state["model_filename"] = model_path.name
                    st.session_state["report_filename"] = report_path.name
            except WorkbenchError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.exception(exc)

        result = st.session_state.get("training_result")
        if result is not None:
            st.markdown("---")
            st.markdown(f"### 4. Experiment results — {result.algorithm_name}")
            _show_metrics(result.metrics)
            st.markdown("#### Output preview")
            st.dataframe(result.output_frame.head(100), use_container_width=True)
            c1, c2 = st.columns(2)
            if st.session_state.get("model_download"):
                c1.download_button("Download trained model (.joblib)", data=st.session_state["model_download"],
                                   file_name=st.session_state["model_filename"], mime="application/octet-stream", use_container_width=True)
            if st.session_state.get("report_download"):
                c2.download_button("Download report (.json)", data=st.session_state["report_download"],
                                   file_name=st.session_state["report_filename"], mime="application/json", use_container_width=True)
            csv_bytes = result.output_frame.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions / transformed output (.csv)", data=csv_bytes,
                               file_name="experiment_output.csv", mime="text/csv")

        if task in {"classification", "regression"}:
            with st.expander("Optional: compare several algorithms using cross-validation"):
                candidates = st.multiselect("Candidate algorithms", options=list(algorithm_map), default=list(algorithm_map)[:min(3, len(algorithm_map))], key="compare_candidates")
                cv = st.slider("CV folds", min_value=2, max_value=10, value=5, key="cv_folds")
                if st.button("Compare candidates", key="compare_button"):
                    try:
                        ids = [algorithm_map[label] for label in candidates]
                        with st.spinner("Running cross-validation on the training portion..."):
                            comparison = compare_models(data, ids, target=target, test_size=test_size, cv=cv, random_state=int(random_state))
                            st.session_state["comparison_result"] = comparison
                    except WorkbenchError as exc:
                        st.error(str(exc))
                if st.session_state.get("comparison_result") is not None:
                    st.dataframe(st.session_state["comparison_result"], use_container_width=True, hide_index=True)
                    st.caption("Model comparison uses CV only on the training partition. The held-out test partition is not used to rank candidates.")

    with inference_tab:
        st.markdown("### Predict with a previously trained supervised model")
        result = st.session_state.get("training_result")
        if result is None or result.task not in {"classification", "regression"}:
            st.info("Train a classification or regression model first. Prediction is not enabled for clustering / dimensionality-reduction artifacts in this prototype.")
        else:
            st.write(f"Active model: **{result.algorithm_name}**")
            expected_features = result.metadata["feature_columns"]
            st.caption("Upload a new CSV/XLSX with the same feature columns used during training. The target column must not be included.")
            predict_file = st.file_uploader("Upload new observations", type=["csv", "xlsx"], key="prediction_upload")
            if predict_file is not None and st.button("Generate predictions", type="primary"):
                try:
                    new_data = validate_dataframe(load_dataframe(predict_file.getvalue(), predict_file.name))
                    missing = sorted(set(expected_features) - set(new_data.columns))
                    extra = sorted(set(new_data.columns) - set(expected_features))
                    if missing:
                        st.error(f"Missing feature columns: {missing}")
                    else:
                        predictions = result.estimator.predict(new_data[expected_features])
                        predicted = new_data.copy()
                        predicted["prediction"] = predictions
                        st.dataframe(predicted.head(100), use_container_width=True)
                        st.download_button("Download prediction results (.csv)", predicted.to_csv(index=False).encode("utf-8"),
                                            file_name="new_data_predictions.csv", mime="text/csv")
                        if extra:
                            st.warning(f"Extra columns were ignored: {extra}")
                except WorkbenchError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

    with catalog_tab:
        st.markdown("### Classical tabular algorithm catalog")
        catalog = pd.DataFrame([{
            "ID": a.id, "Algorithm": a.name, "Task": TASK_LABELS[a.task],
            "Family": a.family, "Description": a.description,
        } for a in get_algorithms()])
        st.dataframe(catalog, use_container_width=True, hide_index=True)
        st.caption("Katalog berisi 40 algoritma scikit-learn dari guidebook. PyTorch/TensorFlow image, text, dan sequence architectures tidak dimasukkan ke tabular prototype ini.")
