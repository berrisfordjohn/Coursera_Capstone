#  Script to compare neighborhoods from two city locations 

# import package
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
import logging  # logging module
import argparse
import matplotlib.pyplot as plt
# from bokeh.io import export_png, export_svgs
# from bokeh.models import ColumnDataSource, DataTable, TableColumn

# credentials are stored in a separate file and not committed
from credentials import CLIENT_ID, CLIENT_SECRET, VERSION, LIMIT

logger = logging.getLogger()


def return_most_common_venues(row, num_top_venues):
    """
    Function to sort venues in descending order
    :param row: Pandas row
    :param num_top_venues: Number of rows to sort by
    :return: 
    """
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]


# def save_df_as_image(df, path):
#     """
#     Save a DataFrame as a nice image
#     :param df: The dataFrame
#     :param path: Filename to save to
#     :return:
#     """
#     source = ColumnDataSource(df)
#     df_columns = [df.index.name]
#     df_columns.extend(df.columns.values)
#     columns_for_table = []
#     for column in df_columns:
#         columns_for_table.append(TableColumn(field=column, title=column))
#
#     data_table = DataTable(source=source, columns=columns_for_table, height_policy="auto", width_policy="auto",
#                            index_position=None)
#     export_png(data_table, filename=path)


class ProcessLocation:
    """
    Class to process location and save a map of the resulting clusters 
    """

    def __init__(self, location, kclusters=10, num_top_venues=5):
        """

        :param str location: location to get information about
        :param int kclusters: Number of clusters
        :param int num_top_venues: Number of venues to group
        """
        self.location = location
        self.coordinate_data = {}
        self.kclusters = kclusters
        self.num_top_venues = num_top_venues
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
        """
        read local coordinate geospatial CSV file and write data into a dictionary 
        :return dict: dictionary of postal codes and coordinates
        """
        with open('Geospatial_Coordinates.csv') as in_file:
            data = csv.DictReader(in_file)
            for row in data:
                self.coordinate_data[row['Postal Code']] = {'longitude': row['Longitude'],
                                                            'latitude': row['Latitude']}
        return self.coordinate_data

    def set_rainbow(self):
        """
        set the colour scheme for the clusters
        :return: 
        """
        x = np.arange(self.kclusters)
        ys = [i + x + (i * x) ** 2 for i in range(self.kclusters)]
        colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
        self.rainbow = [colors.rgb2hex(i) for i in colors_array]

    def get_coordinates(self, postal_code):
        """
        get the longitude and latitude for a given postal code
        :param str postal_code: postal code
        :return: longitude and latitude or None, None is postal code is not present
        """
        ret = self.coordinate_data.get(postal_code, {})
        latitude = ret.get('latitude')
        longitude = ret.get('longitude')
        return longitude, latitude

    def get_data_from_wikipedia(self):
        """
        parse wikipedia page for a given URL - set by self.get_url method
        :return: 
        """
        self.get_url()
        if self.url:
            req = requests.get(self.url)
            soup = BeautifulSoup(req.content, 'html.parser')
            # logging.info(soup.prettify())
            table = soup.find('table', attrs={'class': 'wikitable sortable'})
            table_body = table.find('tbody')
            # logging.info(table_body)

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
        """
        Loads data from wikipedia into a Pandas dataframe
        :return: 
        """
        if self.data_from_wikipedia:
            self.df = pd.DataFrame(self.data_from_wikipedia)
            # rename the postal code heading
            self.df.rename(columns={"Postal Code": "PostalCode"}, inplace=True)

    def add_coordinates(self):
        """
        Adds coordinates (longitude, latitude) to data from wikipedia
        :return: 
        """
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
        """
        Get nearby venues from Foursquare
        :param int radius: radius to get nearby venues
        :return: 
        """
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
            # logging.info(url)
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
        """
        Classifies venues and returns a grouped dataFrame
        :return: 
        """
        logging.info('There are {} uniques categories.'.format(len(self.nearby_venues['Venue Category'].unique())))
        temp_nearby_venues = self.nearby_venues
        temp_nearby_venues['count'] = np.zeros(len(temp_nearby_venues))
        venue_counts = \
        temp_nearby_venues[['Neighborhood', 'Venue Category', 'count']].groupby(['Neighborhood', 'Venue Category'])[
            'count'].count()
        self.grouped_df = venue_counts.unstack(level=-1)
        self.grouped_df = self.grouped_df.fillna(0)
        self.grouped_df = self.grouped_df.reset_index()

    def sort_top_ten_venues(self):
        """
        sorts the venues into
        :return:
        """
        logging.info('sort top ten venues')
        # new dataframe and display the top 10 venues for each neighborhood.

        # create columns according to number of top venues
        columns = ['Neighborhood']
        for ind in np.arange(self.num_top_venues):
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
                                                                                       self.num_top_venues)

        logging.info(self.neighborhoods_venues_sorted)

    def cluster_data(self):
        """
        Clusters data using Kmeans with self.kclusters number of clusters
        :return:
        """
        logging.info('cluster data')
        # set number of clusters
        grouped_clustering = self.grouped_df.drop('Neighborhood', axis=1)

        # run k-means clustering
        kmeans = KMeans(n_clusters=self.kclusters, random_state=0).fit(grouped_clustering)

        # add clustering labels
        self.grouped_df.insert(0, 'Cluster Labels', kmeans.labels_)

        self.clusters_merged = self.df

        # get the longitude and latitude back
        self.clusters_merged = self.clusters_merged.join(self.grouped_df.set_index('Neighborhood'),
                                                         on='Neighborhood')

        self.clusters_merged = self.clusters_merged.dropna()
        self.clusters_merged['Cluster Labels'] = self.clusters_merged['Cluster Labels'].astype('int')
        logging.info(self.clusters_merged)

    def plot_cluster_counts(self):
        """
        get the first set of clusters and plot them
        :return:
        """
        logging.info('get counts of outlets per cluster')
        logging.info(self.grouped_df)
        self.grouped_df['total'] = self.grouped_df.sum(axis=1)

        logging.info('plot cluster counts')
        self.grouped_df.plot.scatter(x='Cluster Labels', y='total')
        plt.title('Number of food outlets per cluster in {}'.format(self.location))
        plt.savefig('{}_cluster_counts'.format(self.location))

        logging.info('plot cluster box plots')
        self.grouped_df[['Cluster Labels', 'total']].boxplot(by='Cluster Labels')
        plt.title('Number of food outlets per cluster in {}'.format(self.location))
        plt.savefig('{}_cluster_counts_box'.format(self.location))

    def plot_cluster_counts_per_type(self, food_outlet):
        """
        Plot clusters and food_outlet
        :return:
        """
        logging.info('get counts of outlets per cluster')
        logging.info(self.grouped_df)

        logging.info('plot cluster box plots')
        self.grouped_df[['Cluster Labels', food_outlet]].boxplot(by='Cluster Labels')
        plt.title('Number of {} outlets per cluster in {}'.format(food_outlet, self.location))
        plt.savefig('{}_{}_cluster_counts_box'.format(self.location, food_outlet))

    def get_highest_count_outlet_per_cluster(self):
        """
        Get the most popular food outlet per cluster
        :return:
        """
        # group by cluster labels and find the average number of outlets per type
        df = self.grouped_df.drop(['Neighborhood', 'total'], axis=1).groupby('Cluster Labels').mean()
        # find the outlet type with the highest count
        df['Most Popular Outlet'] = df.idxmax(axis=1)
        df = df.reset_index()

        # output the data frame
        df = df[['Most Popular Outlet', 'Cluster Labels']]
        df.to_csv('{}_most_popular_outlets.csv'.format(self.location))

    def get_average_latitude_longitude(self):
        """
        For initiating a map a single latitude / longitude is needed
        this method returns the average latitude / longitude from a dataframe
        :return:
        """
        average_latitude = sum(self.df['Latitude']) / len(self.df['Latitude'])
        average_longitude = sum(self.df['Longitude']) / len(self.df['Longitude'])
        return [average_latitude, average_longitude]

    def plot_clusters(self):
        """
        plots clusters on a folium map
        :return:
        """
        # create map
        self.map_clusters = folium.Map(location=self.get_average_latitude_longitude(), zoom_start=11)

        markers_colors = []
        for index, row in self.clusters_merged.iterrows():
            postal_code = row['Borough']
            lat = row['Latitude']
            lon = row['Longitude']
            neighbour = row['Neighborhood']
            cluster = row['Cluster Labels']
            label = folium.Popup('{}-{}, cluster {}'.format(postal_code, neighbour, cluster), parse_html=True)
            folium.CircleMarker(
                [lat, lon],
                radius=5,
                popup=label,
                color=self.rainbow[cluster - 1],
                fill=True,
                fill_color=self.rainbow[cluster - 1],
                fill_opacity=0.7).add_to(self.map_clusters)

    def save_map(self):
        """
        Saves the map
        :return:
        """
        filename = '{}_map.html'.format(self.location)
        self.map_clusters.save(filename)

    def check_local_location_file(self):
        """
        Get location data from local file - used if Borough information is not available in wikipedia
        :return:
        """
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
        """
        get data location data from either a local file or from wikipedia
        :return:
        """
        logging.info('get data for location')
        dataframe_pickle = '{}_dataframe.pkl'.format(self.location)
        if os.path.exists(dataframe_pickle):
            self.df = pd.read_pickle(dataframe_pickle)
        else:
            if not self.check_local_location_file():
                self.get_data_from_wikipedia()
                self.load_data_into_dataframe()
                self.add_coordinates()
            self.df.to_pickle(dataframe_pickle)
        logging.info(self.df.head())

    def get_data_for_nearby_venues(self):
        """
        get data for nearby venues
        :return:
        """
        logging.info('get nearby venues')
        dataframe_pickle = '{}_nearby_veneues.pkl'.format(self.location)
        if os.path.exists(dataframe_pickle):
            self.nearby_venues = pd.read_pickle(dataframe_pickle)
        else:
            self.get_nearby_venues()
            self.nearby_venues.to_pickle(dataframe_pickle)

    def process_url(self):
        """
        get location data, get nearby venues and classify them
        :return:
        """

        self.get_data_for_location()
        self.get_data_for_nearby_venues()
        self.classification_of_venues()

    def get_url(self):
        """
        set self.url if the location is known
        :return: url or None
        """
        location_url = {'toronto': 'https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M',
                        }
        self.url = location_url.get(self.location.lower())
        return self.url

    def run_process(self):
        """
        main method to process the location
        :return:
        """
        # get data for the location set when initiating the class
        self.process_url()
        # sort the venues into top ten venues
        # self.sort_top_ten_venues()
        # cluster the data
        self.cluster_data()

        self.plot_cluster_counts()
        self.get_highest_count_outlet_per_cluster()
        self.plot_cluster_counts_per_type(food_outlet='Pizza Place')
        self.plot_cluster_counts_per_type(food_outlet='Café')
        self.plot_cluster_counts_per_type(food_outlet='Vietnamese Restaurant')

        # plot clusters
        self.plot_clusters()
        # save plotted clusters
        self.save_map()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--debug', help='debugging', action='store_const', dest='loglevel', const=logging.info,
                        default=logging.INFO)
    parser.add_argument('--locations', help='locations to run', type=str, default='toronto,newyork')

    args = parser.parse_args()

    logger.setLevel(args.loglevel)

    # runs the process on two locations
    for location in args.locations.split(','):
        pl = ProcessLocation(location)
        pl.run_process()


if __name__ == '__main__':
    main()
