# ClassSense

A classroom environmental monitoring prototype built with Python, Streamlit, and Isolation Forest.

ClassSense explores how machine learning-based anomaly detection compares with traditional threshold-based monitoring using simulated classroom temperature, humidity, and lighting data.

### Live Demo

**[Try ClassSense on Streamlit](https://classsense-iot.streamlit.app/)**

Explore the interactive dashboard by adjusting simulated classroom conditions and observing how threshold-based rules and the Isolation Forest model respond to different readings.

### Features

- Interactive temperature, humidity, and illumination simulation
- Threshold-based environmental alerts
- Isolation Forest anomaly detection
- Performance comparison using synthetic test data
- Suggested environmental checks based on simulated conditions

### Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas and NumPy

### Project Scope

ClassSense is an exploratory prototype developed to examine the potential of IoT-assisted classroom monitoring. The dashboard currently uses synthetic data rather than live sensor measurements.

Virtual sensor acquisition was explored separately using ESP32 and Wokwi. Real-time integration between the sensor simulation and dashboard is outside the current implementation.
