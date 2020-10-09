from flask import Flask, render_template, request
import plotly
import plotly.graph_objs as go
import plotly.express as px
import json
import io
import base64
import sys
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# sys.path.append('/Desktop/myproject')
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)

sns.set_style(style='dark')
sns.color_palette("rocket")







@app.route("/")
def homepage():

    return 'Go to /home'

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/home')
def imlucky():
    data = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/worldwide-aggregate.csv')
    deaths = np.array(data['Deaths'].values)
    confirmed = np.array(data['Confirmed'].values)
    basic_data = { 'dead' :deaths[-1], 'conf' : confirmed[-1] }
    graph_data = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')
    graph_data['Date'] = pd.to_datetime(graph_data['Date'].astype(str), format = '%Y-%m-%d')
    
    global_graph = graph_data.groupby(['Date'])['Confirmed','Recovered','Deaths'].sum()
    global_graph.reset_index(inplace = True)
    global_graph[['Confirmed','Recovered','Deaths']] = (global_graph[['Confirmed','Recovered','Deaths']].values)/1e6
    

    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Global Spread of the Virus")
    axis.set_xlabel("Time")
    axis.set_ylabel("Cases (in mil)")
    sns.lineplot(x = 'Date', y = 'Confirmed' , data = global_graph, ax =axis)
    sns.lineplot(x = 'Date', y = 'Deaths' , data = global_graph, ax =axis, color = 'red')
    sns.lineplot(x = 'Date', y = 'Recovered' , data = global_graph, ax =axis, color = 'green')
    axis.legend(['confirmed', 'deaths', 'recovered'], loc = 'upper left')

    #Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')


    top_c = ['US', 'India', 'Brazil', 'Russia', 'Colombia', 'Peru', 'Mexico',
       'South Africa', 'Spain', 'Argentina']


    countries = graph_data.groupby(['Date', 'Country'])['Confirmed','Recovered','Deaths'].sum()
    countries.reset_index(inplace = True)

    countries['Country'] = [x if x in top_c else 'Other' for x in countries['Country']]
    top_countries = countries[countries['Country']!= 'Other']

    top_countries[['Confirmed','Recovered','Deaths']] = (top_countries[['Confirmed','Recovered','Deaths']].values)/1e5

    fig1 = Figure()
    axis = fig1.add_subplot(1, 1, 1)
    axis.set_title("Country-wise Spread of the Virus")
    axis.set_xlabel("Time")
    axis.set_ylabel("Cases (in 100k)")
    axis.legend(loc = 'upper left')
    sns.lineplot(x = 'Date', y = 'Confirmed' , hue = 'Country', hue_order = top_c, palette = 'husl', data = top_countries, estimator = 'sum', ax = axis)
    
    #Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig1).print_png(pngImage)

    # Encode PNG image to base64 string
    pngImageB64String1 = "data:image/png;base64,"
    pngImageB64String1 += base64.b64encode(pngImage.getvalue()).decode('utf8')

    bar, scatter = create_plot('conf')
          
    return render_template('index.html', basic = basic_data,  image = pngImageB64String, image1 = pngImageB64String1, plot = {'data': bar,'layout': scatter })


def create_plot(feature):
    counties = json.load(open('world-countries.json', 'r'))

    graph_data = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv')
    graph_data['Date'] = pd.to_datetime(graph_data['Date'].astype(str), format = '%Y-%m-%d')
    df = graph_data[graph_data['Date']== max(graph_data['Date'])]
    df['Country'] = df['Country/Region']
    df = df.drop(['Province/State', 'Country/Region'], axis = 1)

    state_map_id = {}
    for feature in counties['features']:
        feature['iso_code'] = feature['id']
        state_map_id[feature['properties']['name']] = feature['iso_code']
    
    df['iso_code'] = [lambda x : state_map_id[x] if x in df['Country'] else None for x in df['Country']]
    
    if feature == 'conf':
        data = [px.choropleth(
            data_frame=df, locations='Country', locationmode='country names',
            geojson=counties, featureidkey='iso_code' , color='Confirmed', color_continuous_scale=px.colors.sequential.Plasma,
            projection='equirectangular' )
        ]
                
    elif feature == 'dead':
        data = [px.choropleth(
            data_frame=df, locations='Country', locationmode='country names',
            geojson=counties, featureidkey='iso_code' , color='Deaths', color_continuous_scale=px.colors.sequential.Plasma,
            projection='equirectangular' )
        ]
    else:
        fig = px.choropleth(
            data_frame=df, locations='Country', locationmode='country names',
            geojson=counties, featureidkey='iso_code' , color='Deaths', color_continuous_scale="Viridis",
            projection='equirectangular' )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        data = [fig]
    dataJSON = data[0]['data']
    layoutJSON = data[0]['layout']
        
    dataJSON = json.dumps(dataJSON, cls=plotly.utils.PlotlyJSONEncoder)
    layoutJSON = json.dumps(layoutJSON, cls=plotly.utils.PlotlyJSONEncoder)

    return dataJSON, layoutJSON



@app.route('/bar', methods=['GET', 'POST'])
def change_features():

    feature = request.args.get('selected')
    dataJSON, layoutJSON = create_plot(feature)

    return dataJSON, layoutJSON

if '__main__' == __name__ :
    app.run(host= 'localhost', port=8000, debug= True)
