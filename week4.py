import os
import pandas as pd  # library for data analysis
from bs4 import BeautifulSoup  # library to parse web pages
import requests  # library to handle requests
import csv
import folium  # map rendering library
from sklearn.cluster import KMeans
import numpy as np
# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors
import json
from credentials import CLIENT_ID, CLIENT_SECRET, VERSION, LIMIT


# function to sort the venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]


class ProcessLocation:

    def __init__(self, location):
        self.location = location
        self.coordinate_data = {}
        self.kclusters = 10
        self.rainbow = []
        self.url = None
        self.longitude = []
        self.latitude = []
        self.data_from_wikipedia = []
        self.df = None
        self.grouped_df = None
        self.nearby_venues = None
        self.neighborhoods_venues_sorted = None
        self.map_clusters = None
        self.read_local_coordinates_file()
        self.set_rainbow()

    def read_local_coordinates_file(self):
        with open('Geospatial_Coordinates.csv') as in_file:
            data = csv.DictReader(in_file)
            for row in data:
                self.coordinate_data[row['Postal Code']] = {'longitude': row['Longitude'],
                                                            'latitude': row['Latitude']}

    def set_rainbow(self):
        # set color scheme for the clusters
        x = np.arange(self.kclusters)
        ys = [i + x + (i * x) ** 2 for i in range(self.kclusters)]
        colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
        self.rainbow = [colors.rgb2hex(i) for i in colors_array]

    def get_coordinates(self, postal_code):
        ret = self.coordinate_data.get(postal_code, {})
        latitude = ret.get('latitude')
        longitude = ret.get('longitude')
        return longitude, latitude

    def get_data_from_wikipedia(self):
        self.get_url()
        if self.url:
            req = requests.get(self.url)
            soup = BeautifulSoup(req.content, 'html.parser')
            # print(soup.prettify())
            table = soup.find('table', attrs={'class': 'wikitable sortable'})
            table_body = table.find('tbody')
            # print(table_body)

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
        if self.data_from_wikipedia:
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

    def get_nearby_venues(self, radius=500):
        names = self.df['Neighborhood']
        latitudes = self.df['Latitude']
        longitudes = self.df['Longitude']
        venues_list = []
        for name, lat, lng in zip(names, latitudes, longitudes):
            url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}&section=food'.format(
                CLIENT_ID,
                CLIENT_SECRET,
                VERSION,
                lat,
                lng,
                radius,
                LIMIT)
            # print(url)
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

    def classification_of_venues(self):
        print('There are {} uniques categories.'.format(len(self.nearby_venues['Venue Category'].unique())))
        temp_nearby_venues = self.nearby_venues
        temp_nearby_venues['count'] = np.zeros(len(temp_nearby_venues))
        venue_counts = temp_nearby_venues.groupby(['Neighborhood', 'Venue Category']).count()
        print(venue_counts[(venue_counts['count'] > 2)])
        onehot = pd.get_dummies(self.nearby_venues[['Venue Category']], prefix="", prefix_sep="")
        # add neighborhood column back to dataframe
        onehot['Neighborhood'] = self.nearby_venues['Neighborhood']
        print(onehot)

        self.grouped_df = onehot.groupby('Neighborhood').count().reset_index()
        print(self.grouped_df)

    def sort_top_ten_venues(self):
        print('sort top ten venues')
        # new dataframe and display the top 10 venues for each neighborhood.
        num_top_venues = 5

        indicators = ['st', 'nd', 'rd']

        # create columns according to number of top venues
        columns = ['Neighborhood']
        for ind in np.arange(num_top_venues):
            columns.append('{}'.format(ind + 1))
            # try:
            #    columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
            # except:
            #    columns.append('{}th Most Common Venue'.format(ind+1))

        # create a new dataframe
        self.neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
        self.neighborhoods_venues_sorted['Neighborhood'] = self.grouped_df['Neighborhood']

        for ind in np.arange(self.grouped_df.shape[0]):
            self.neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(self.grouped_df.iloc[ind, :],
                                                                                       num_top_venues)

        print(self.neighborhoods_venues_sorted)

    def cluster_data(self):
        print('cluster data')
        # set number of clusters
        grouped_clustering = self.grouped_df.drop('Neighborhood', 1)

        # run k-means clustering
        kmeans = KMeans(n_clusters=self.kclusters, random_state=0).fit(grouped_clustering)

        # add clustering labels
        self.neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

        self.clusters_merged = self.df

        # merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
        self.clusters_merged = self.clusters_merged.join(self.neighborhoods_venues_sorted.set_index('Neighborhood'),
                                                         on='Neighborhood')

        print(self.clusters_merged)

    def get_average_latitude_longitude(self):
        average_latitude = sum(self.df['Latitude']) / len(self.df['Latitude'])
        average_longitude = sum(self.df['Longitude']) / len(self.df['Longitude'])
        return [average_latitude, average_longitude]

    def plot_clusters(self):
        # create map
        self.map_clusters = folium.Map(location=self.get_average_latitude_longitude(), zoom_start=11)

        self.clusters_merged = self.clusters_merged.dropna()
        self.clusters_merged['Cluster Labels'] = self.clusters_merged['Cluster Labels'].astype('int')

        markers_colors = []
        for index, row in self.clusters_merged.iterrows():
            postal_code = row['Borough']
            lat = row['Latitude']
            lon = row['Longitude']
            neighbour = row['Neighborhood']
            cluster = row['Cluster Labels']
            label = folium.Popup(str(postal_code) + ' Cluster ' + str(neighbour), parse_html=True)
            folium.CircleMarker(
                [lat, lon],
                radius=5,
                popup=label,
                color=self.rainbow[cluster - 1],
                fill=True,
                fill_color=self.rainbow[cluster - 1],
                fill_opacity=0.7).add_to(self.map_clusters)

    def save_map(self):
        filename = '{}_map.html'.format(self.location)
        self.map_clusters.save(filename)

    def check_local_location_file(self):
        local_file = '{}_data.json'.format(self.location)
        ret_data = []
        if os.path.exists(local_file):
            with open(local_file) as in_file:
                local_data = json.load(in_file)
                features = local_data.get('features', [])
                for data in features:
                    borough = data['properties']['borough']
                    neighborhood_name = data['properties']['name']

                    neighborhood_latlon = data['geometry']['coordinates']
                    neighborhood_lat = neighborhood_latlon[1]
                    neighborhood_lon = neighborhood_latlon[0]

                    ret_data.append({'Borough': borough,
                                     'Neighborhood': neighborhood_name,
                                     'Latitude': neighborhood_lat,
                                     'Longitude': neighborhood_lon})
        if ret_data:
            self.df = pd.DataFrame(ret_data)
            return True
        return False

    def get_data_for_location(self):
        print('get data for location')
        dataframe_pickle = '{}_dataframe.pkl'.format(self.location)
        if os.path.exists(dataframe_pickle):
            self.df = pd.read_pickle(dataframe_pickle)
        else:
            if not self.check_local_location_file():
                self.get_data_from_wikipedia()
                self.load_data_into_dataframe()
                self.add_coordinates()
            self.df.to_pickle(dataframe_pickle)
        print(self.df.head())

    def get_data_for_nearby_venues(self):
        print('get nearby venues')
        dataframe_pickle = '{}_nearby_veneues.pkl'.format(self.location)
        if os.path.exists(dataframe_pickle):
            self.nearby_venues = pd.read_pickle(dataframe_pickle)
        else:
            self.get_nearby_venues()
            self.nearby_venues.to_pickle(dataframe_pickle)

    def process_url(self):
        self.get_data_for_location()
        self.get_data_for_nearby_venues()
        self.classification_of_venues()

    def run_process(self):
        self.process_url()
        self.sort_top_ten_venues()
        self.cluster_data()
        self.plot_clusters()
        self.save_map()

    def get_url(self):
        location_url = {'toronto': 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M',
                        }
        self.url = location_url.get(self.location.lower())


def main():
    pl = ProcessLocation('newyork')
    pl.run_process()


if __name__ == '__main__':
    main()
