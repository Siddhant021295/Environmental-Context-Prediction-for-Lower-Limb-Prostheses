
from plotly.subplots import make_subplots
import plotly.graph_objects as go

for instance in data_instances:
    data = data_all[data_all['subject_name_instance']==instance]
    fig = make_subplots(rows=3, 
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        x_title='Time (s)',
                        subplot_titles=("Accelerometer","Gyroscope", "Labels"))

    # Adding Gyroscope values
    fig.add_trace(go.Scatter(x=data.timestamp,y=data.x_accelerometer,mode='lines',legendgroup='g',name='Accelerometer X',line=dict(color="Blue")),row=1, col=1)
    fig.add_trace(go.Scatter(x=data.timestamp,y=data.y_accelerometer,mode='lines',legendgroup='g',name='Accelerometer Y',line=dict(color="Red")),row=1, col=1)
    fig.add_trace(go.Scatter(x=data.timestamp,y=data.z_accelerometer,mode='lines',legendgroup='g',name='Accelerometer Z',line=dict(color="Green")),row=1, col=1)


    # Adding Acclerometer values
    fig.add_trace(go.Scatter(x=data.timestamp,y=data.x_gyroscope,mode='lines',legendgroup='g',name='Gyroscope X',line=dict(color="Blue")),row=2, col=1)
    fig.add_trace(go.Scatter(x=data.timestamp,y=data.y_gyroscope,mode='lines',legendgroup='g',name='Gyroscope Y',line=dict(color="Red")),row=2, col=1)
    fig.add_trace(go.Scatter(x=data.timestamp,y=data.z_gyroscope,mode='lines',legendgroup='g',name='Gyroscope Z',line=dict(color="Green")),row=2, col=1)

    #Adding Labels
    fig.add_trace(go.Scatter(x=data.timestamp,y=data.labels,mode='lines',legendgroup='l',name='Groundtruth',line=dict(color="Blue")),row=3, col=1)

    fig.update_yaxes(title_text="w(rad/s)", row=1, col=1)
    fig.update_yaxes(title_text="w(rad/s)", row=2, col=1)
    fig.update_yaxes(title_text="Labels", row=3, col=1)

    fig.update_layout(height=800, width=1000,title_text=instance)
    fig.show()
    fig.write_html('/content/drive/MyDrive/NN project Class/Visualization/'+instance+".html")
