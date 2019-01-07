import React, { Component } from 'react';
import ReactMapGL from 'react-map-gl';
import logo from './logo.svg';
import './App.css';

const MAPBOX_TOKEN = "pk.eyJ1IjoicGhpd2lzc2Vuc2NoYWZ0IiwiYSI6ImM5ZTE5NjMwMjU3MTg0M2RmYWRlNmZmMDVmZDdjOWJlIn0.sQjWofB1mechXvK88OKUqw";

class Map extends Component {

    state = {
        viewport: {
            width: 1600,
            height: 600,
            latitude: 37.7577,
            longitude: -122.4376,
            zoom: 8
        }
    };

    render() {
        return (
            <ReactMapGL
              {...this.state.viewport}
              onViewportChange={(viewport) => this.setState({viewport})}
              mapboxApiAccessToken={MAPBOX_TOKEN}
            />
        );
    }
}


class App extends Component {
    render() {
        return (
            <div className="App">
              <header className="App-header">
                <img src={logo} className="App-logo" alt="logo" />
                <p>
                  Edit <code>src/App.js</code> and save to reload.
                </p>
                <a
                  className="App-link"
                  href="https://reactjs.org"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Learn React
                </a>
              </header>
              <Map />
            </div>
        );
    }
}

export default App;
