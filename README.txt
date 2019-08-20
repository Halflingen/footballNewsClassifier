installs made to make react work from history. Think it is a bit to much:
(update NPM first)
$ npm init -y #this was done in \project
$ npm i webpack webpack-cli --save-dev
change settings in package.jason. add the following
"scripts": {
  "dev": "webpack --mode development ./arx/frontend/src/index.js,
  --output ./arx/frontend/static/frontend/main.js",
  "build": "webpack --mode production ./arx/frontend/src/index.js
  --output ./arx/frontend/static/frontend/main.js",
}

$ npm i @babel/core babel-loader @babel/preset-env @babel/preset-react
  babel-plugin-transform-class-properties --save-dev
$ npm i react react-dom prop-types --save-dev

make .babelrc and webpack.config.js
.babelrc
{
    "presets": [
        "@babel/preset-env", "@babel/preset-react"
    ],
    "plugins": [
        "transform-class-properties"
    ]
}

webpack.config.js
module.exports = {
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader"
        }
      }
    ]
  }
};

$ npm i weak-key --save-dev
$ npm run dev
