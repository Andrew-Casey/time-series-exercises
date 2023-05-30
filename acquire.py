#imports

import pandas as pd
import requests
import os
import io

def fetch_data(url):
    response = requests.get(url)
    data = response.json()
    return data['results']

def get_people():
    # Check if the CSV file exists
    if os.path.exists('people.csv'):
        # If the file exists, load it into a data frame
        people_df = pd.read_csv('people.csv')
    else:
        # Initialize an empty list to store the people data
        all_people = []

        # Fetch data from the first page
        people = requests.get('https://swapi.dev/api/people/')
        data = people.json()

        # Add the people from the first page to the list
        all_people.extend(fetch_data('https://swapi.dev/api/people/'))

        # Check if there are additional pages
        while data['next'] is not None:
            next_page_url = data['next']
            # Fetch data from the next page
            all_people.extend(fetch_data(next_page_url))
            response_next_page = requests.get(next_page_url)
            data = response_next_page.json()

        # Create a data frame from the list of people
        people_df = pd.DataFrame(all_people)

        # Write the data frame to a CSV file
        people_df.to_csv('people.csv', index=False)

    return people_df

def get_planets():
    # Check if the CSV file exists
    if os.path.exists('planets.csv'):
        # If the file exists, load it into a data frame
        planets_df = pd.read_csv('planets.csv')
    else:
        # Initialize an empty list to store the people data
        all_planets = []

        # Fetch data from the first page
        planets = requests.get('https://swapi.dev/api/planets/')
        data = planets.json()

        # Add the people from the first page to the list
        all_planets.extend(fetch_data('https://swapi.dev/api/planets/'))

        # Check if there are additional pages
        while data['next'] is not None:
            next_page_url = data['next']
            # Fetch data from the next page
            all_planets.extend(fetch_data(next_page_url))
            response_next_page = requests.get(next_page_url)
            data = response_next_page.json()

        # Create a data frame from the list of people
        planets_df = pd.DataFrame(all_planets)

        # Write the data frame to a CSV file
        planets_df.to_csv('planets.csv', index=False)

    return planets_df

def get_ships():
    # Check if the CSV file exists
    if os.path.exists('ships.csv'):
        # If the file exists, load it into a data frame
        ships_df = pd.read_csv('ships.csv')
    else:
        # If the file doesn't exist, fetch data and create a data frame
        all_ships = []

        # Fetch data from the first page
        ships = requests.get('https://swapi.dev/api/starships/')
        data = ships.json()

        # Add the ships from the first page to the list
        all_ships.extend(fetch_data('https://swapi.dev/api/starships/'))

        # Check if there are additional pages
        while data['next'] is not None:
            next_page_url = data['next']
            # Fetch data from the next page
            all_ships.extend(fetch_data(next_page_url))
            response_next_page = requests.get(next_page_url)
            data = response_next_page.json()

        # Create a data frame from the list of ships
        ships_df = pd.DataFrame(all_ships)

        # Write the data frame to a CSV file
        ships_df.to_csv('ships.csv', index=False)

    return ships_df

def combine_csv_data():
    # Define the filenames
    people_csv = 'people.csv'
    planets_csv = 'planets.csv'
    ships_csv = 'ships.csv'

    # Read the CSV files into separate data frames
    people_df = pd.read_csv(people_csv)
    planets_df = pd.read_csv(planets_csv)
    ships_df = pd.read_csv(ships_csv)

    # Concatenate the data frames horizontally
    combined_df = pd.concat([people_df, planets_df, ships_df], axis=0)

    return combined_df


    
def get_power(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Read the response content into a DataFrame
    df = pd.read_csv(io.StringIO(response.text))

    return df
  
