"""
Script for testing a deployed API by sending a sample request and checking the response.

The script sends a sample data input to a specified API endpoint and checks if the response status code is 200.
It logs information about the status code, response content, and the time taken for the request.

"""

import requests
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


def api_test(url, data_input):
    """
    Test the deployed API by sending a POST request and checking the response.

    Parameters:
    - url (str): The URL of the API endpoint.
    - data_input (dict): The input data to be sent with the POST request.

    Returns:
    None

    Raises:
    - requests.exceptions.RequestException: If the request fails.

    """
    try:
        start_time = time.time()
        resp = requests.post(url, json=data_input)

        # check if the request was successful
        resp.raise_for_status()

        elapsed_time = time.time() - start_time

        # Log the response content for better understanding
        logging.info("Response Content:")
        logging.info(resp.text)
        logging.info("Prediction")
        logging.info(resp.json()[0]["prediction"])

        logging.info("Testing Render app")
        logging.info(f"Status code: {resp.status_code}")
        logging.info(f"Time taken: {elapsed_time:.4f} seconds")

    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}")


if __name__ == "__main__":
    # Update the URL to your specific API endpoint
    api_url = "https://udacity-app-nbh.onrender.com/predict"

    data_input = {
        "age": 53,
        "workclass": "Private",
        "fnlgt": 169846,
        "education": "HS-grad",
        "education_num": 9,
        "marital_status": "Married-civ-spouse",
        "occupation": "Adm-clerical",
        "relationship": "Wife",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }

    api_test(api_url, data_input)
