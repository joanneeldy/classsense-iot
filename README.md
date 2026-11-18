# ClassSense

ClassSense is a Streamlit prototype for exploring classroom environmental monitoring. It compares basic temperature, humidity and light thresholds with an Isolation Forest anomaly detector.

The example is based on a classroom with two AC units, two ceiling fans and manually switched lighting. All readings in the app are **simulated**; it is not connected to physical sensors or appliances.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## How it works

- Adjust temperature, humidity and illuminance in the scenario simulator.
- Compare the threshold decision with the ML anomaly flag for the same reading.
- Review evaluation metrics on a separate synthetic test set.
- Download evaluation results or synthetic test data as CSV.

The model is trained on 700 assumed-normal synthetic examples. Evaluation uses 240 synthetic normal cases and 60 injected anomalies, with a fixed random seed for repeatable results.

## Limitations

Synthetic results are not evidence of real-world accuracy or classroom comfort. The anomaly score is not a probability. Proposed sensor integration, validation using real measurements, and secure telemetry remain future work.

## Deployment

Deploy using [Streamlit Community Cloud](https://share.streamlit.io/) and select `app.py` as the main file. Dependencies are listed in `requirements.txt`.
