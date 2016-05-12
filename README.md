# Digitize
An Android app that recognizes handwritten digits

This app lets users take a picture of a hand written digit. This picture is then send to the server. The server will put it through a neuarl network. The neural network will recognize the digit and send the result back to the user.

The app uses OKHTTP on the client side to send the data to the server. The server is written in Python and uses Flask. You can find the server code and the Neural Network code in the "engineapp" folder.

The app is not avaliple at the play store yet, I need to make few adjustments to prevent it from crashing.
