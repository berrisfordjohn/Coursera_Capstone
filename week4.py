import os
import pandas as pd # library for data analysis
from bs4 import BeautifulSoup # library to parse web pages
import requests # library to handle requests
import csv
import folium # map rendering library
from sklearn.cluster import KMeans
import numpy as np
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
from credentials import CLIENT_ID, CLIENT_SECRET, VERSION, LIMIT


# function to sort the venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]


class ProcessLocation:

    def __init__(self):
        self.coordinate_data = {}
        self.longitude = []
        self.latitude = []
        self.data_from_wikipedia = []
        self.nearby_venues = None
        self.neighborhoods_venues_sorted = None
        self.read_local_coordinates_file()

    def read_local_coordinates_file(self):
        with open('Geospatial_Coordinates.csv') as in_file:
            data = csv.DictReader(in_file)
            for row in data:
                self.coordinate_data[row['Postal Code']] = {'longitude': row['Longitude'],
                                                    'latitude': row['Latitude']}

    def get_coordinates(self, postal_code):
        ret = self.coordinate_data.get(postal_code, {})
        latitude = ret.get('latitude')
        longitude = ret.get('longitude')
        return longitude, latitude

    def get_data_from_wikipedia(self, url):
        req = requests.get(url)
        soup = BeautifulSoup(req.content, 'html.parser')
        #print(soup.prettify())
        table = soup.find('table', attrs={'class':'wikitable sortable'})
        table_body = table.find('tbody')
        #print(table_body)

        # get the headers of the table and store in a list
        table_headers = []
        headers = table_body.find_all('th')
        for header in headers:
            header_value = header.get_text().strip()
            table_headers.append(header_value)

        # get the rows of the table
        rows = table_body.find_all('tr')
        for row in rows:
            row_data = {}
            cells = row.find_all('td')
            for position, cell in enumerate(cells):
                value = cell.get_text().strip()
                key = table_headers[position]
                # add the value to a dictionary
                row_data[key] = value

            # check that there is some data and that Borough is not unassigned
            if row_data and row_data.get('Borough', '') != 'Not assigned':
                if 'Neighbourhood' in row_data:
                    row_data['Neighborhood'] = row_data.pop('Neighbourhood')
                self.data_from_wikipedia.append(row_data)


    def load_data_into_dataframe(self):
        self.df = pd.DataFrame(self.data_from_wikipedia)
        # rename the postal code heading
        self.df.rename(columns={"Postal Code": "PostalCode"}, inplace=True)

    def add_coordinates(self):
        self.longitude = []
        self.latitude = []

        for index, row in self.df.iterrows():
            postal_code = row.get('PostalCode')
            row_long, row_lat = self.get_coordinates(postal_code=postal_code)
            self.longitude.append(float(row_long))
            self.latitude.append(float(row_lat))

        self.df['Latitude'] = self.latitude
        self.df['Longitude'] = self.longitude

        return self.df

    def getNearbyVenues(self, radius=500):
        names = self.df['Neighborhood']
        latitudes = self.df['Latitude']
        longitudes = self.df['Longitude']
        venues_list=[]
        for name, lat, lng in zip(names, latitudes, longitudes):
            url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
                CLIENT_ID,
                CLIENT_SECRET,
                VERSION,
                lat,
                lng,
                radius,
                LIMIT)
            results = requests.get(url).json()["response"]['groups'][0]['items']
            venues_list.append([(
                name,
                lat,
                lng,
                v['venue']['name'],
                v['venue']['location']['lat'],
                v['venue']['location']['lng'],
                v['venue']['categories'][0]['name']) for v in results])

        self.nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
        self.nearby_venues.columns = ['Neighborhood',
                    'Neighborhood Latitude',
                    'Neighborhood Longitude',
                    'Venue',
                    'Venue Latitude',
                    'Venue Longitude',
                    'Venue Category']


    def process_url(self, url):
        self.get_data_from_wikipedia(url=url)
        self.load_data_into_dataframe()
        self.add_coordinates()
        self.getNearbyVenues()
        print('There are {} uniques categories.'.format(len(self.nearby_venues['Venue Category'].unique())))
        temp_nearby_venues = self.nearby_venues
        temp_nearby_venues['count'] = np.zeros(len(temp_nearby_venues))
        venue_counts = temp_nearby_venues.groupby(['Neighborhood', 'Venue Category']).count()
        print(venue_counts[(venue_counts['count'] > 2)])
        onehot = pd.get_dummies(self.nearby_venues[['Venue Category']], prefix="", prefix_sep="")
        # add neighborhood column back to dataframe
        onehot['Neighborhood'] = self.nearby_venues['Neighborhood']

        self.grouped_df = onehot.groupby('Neighborhood').mean().reset_index()
        print(self.grouped_df.head())

    def sort_top_ten_venues(self):
        print('sort top ten venues')
        # new dataframe and display the top 10 venues for each neighborhood.
        num_top_venues = 10

        indicators = ['st', 'nd', 'rd']

        # create columns according to number of top venues
        columns = ['Neighborhood']
        for ind in np.arange(num_top_venues):
            try:
                columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
            except:
                columns.append('{}th Most Common Venue'.format(ind+1))

        # create a new dataframe
        self.neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
        self.neighborhoods_venues_sorted['Neighborhood'] = self.grouped_df['Neighborhood']

        for ind in np.arange(self.grouped_df.shape[0]):
            self.neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(self.grouped_df.iloc[ind, :], num_top_venues)

        print(self.neighborhoods_venues_sorted.head())


    def cluster_data(self):
        # set number of clusters
        kclusters = 5

        grouped_clustering = grouped.drop('Neighborhood', 1)

        # run k-means clustering
        kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(grouped_clustering)

        # check cluster labels generated for each row in the dataframe
        kmeans.labels_[0:10]

        # add clustering labels
        neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

        toronto_merged = df

        # merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
        toronto_merged = toronto_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

        toronto_merged.head()

        # set color scheme for the clusters
        x = np.arange(kclusters)
        ys = [i + x + (i*x)**2 for i in range(kclusters)]
        colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
        rainbow = [colors.rgb2hex(i) for i in colors_array]

    def save_pickle_top_ten(self, filename):
        self.neighborhoods_venues_sorted.to_pickle(filename)

    def read_pickle(self, filename):
        if os.path.exists(filename):
            self.neighborhoods_venues_sorted = pd.read_pickle(filename)


    def run_process(self, url, filename):
        if os.path.exists(filename):
            self.read_pickle(filename)
        else:
            self.process_url(url=url)
            self.sort_top_ten_venues()
            self.save_pickle_top_ten(filename=filename)


    def get_data_for_toronto(self):
        filename = 'toronto.pkl'
        url = 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M'
        self.run_process(url=url, filename=filename)



def main():
    pl = ProcessLocation()
    pl.get_data_for_toronto()

if __name__ == '__main__':
    main()    
