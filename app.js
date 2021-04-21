/*
Copyright (c) 2021 Oliver Sharif

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is furnished
to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

 */


// Create app
var app = angular.module('app', []);

// Create Controller
app.controller('main', ['$scope', '$document', '$log', function($scope, $document, $log){
  // Lifesigns
  $log.log('Angular Ready!');
  // Tensorflow
  if (tf !== null){
    $log.log("Tensorflow Ready!");
  }
  // OpenCV
  if (cv !== null){
    $log.log("OpenCV Ready!");
  }
  // Lifesigns End

  // Vars
  $scope.digits = [0,2,3,4,5,6,8];
  $scope.zahl = -1;
  $scope.m = false;
  $scope.drawing = false;
  $scope.canvas = $document[0].getElementById('lcanvas');
  $scope.dstCanvas = $document[0].getElementById('dstCanvas');
  $scope.ctx = $scope.canvas.getContext('2d');
  $scope.ctx.fillStyle = "white";
  $scope.ctx.fillRect(0, 0, 224, 224);

  /*FUNCTIONS*/
  //startpaint
  $scope.startpaint = function(e){
    $scope.getPos(e);
    $scope.drawing = true;
  }

  //stoppaint
  $scope.stoppaint = function(e){
    $scope.drawing = false;
  }

  $scope.paint = function(e){
    if ($scope.drawing){
        $scope.ctx.beginPath();
        $scope.ctx.lineWidth = 20;
        $scope.ctx.lineCap = "round";
        $scope.ctx.strokeStyle = "black";
        $scope.ctx.moveTo($scope.X, $scope.Y);
        $scope.getPos(e);
        $scope.ctx.lineTo($scope.X, $scope.Y);
        $scope.ctx.stroke();
    }
  }

  $scope.getPos = function(e){
    $scope.X = e.clientX - $scope.canvas.offsetLeft;
    $scope.Y = e.clientY - $scope.canvas.offsetTop;
  }

  //if pointer Leaves input Field
  $scope.leave = function(){
    $scope.drawing = false;
  }

  // Work on image
  $scope.workImage = function(){
    let src = cv.imread('lcanvas');
    let dst = new cv.Mat();
    cv.threshold(src, dst, 177, 255, cv.THRESH_BINARY_INV);
    let last = new cv.Mat();
    $scope.minAreaRect(dst);
    dst = src.roi($scope.minAreaRect(dst));
    cv.resize(dst, dst, new cv.Size(224, 224), 0,0, cv.INTER_AREA);
    cv.imshow('dstCanvas', dst);
  }

  // get the minimum BLACK RECT
  $scope.minAreaRect = function(img){
    let x = 255, y = 255, x2 = 0, y2 = 0;
    for(let i = 0; i < 224; i++){
      for(let j = 0; j < 224; j++){
        y = (i < y && img.ucharPtr(i, j)[0] == 255)?i:y;
        x = (j < x && img.ucharPtr(i, j)[0] == 255)?j:x;
        y2 = (i > y2 && img.ucharPtr(i, j)[0] == 255)?i:y2;
        x2 = (j > x2 && img.ucharPtr(i, j)[0] == 255)?j:x2;
      }
    }
    return new cv.Rect(x, y, x2 - x, y2 - y);
  }

  //Load Tensorflow Model
  $scope.loadModel = function(){
    tf.loadLayersModel("my-model.json").then((model) =>{
      $scope.model = model;
      $scope.m = true;
    });
  }

  // Do Prediction
  $scope.predict = function(){
    let result = $scope.model.predict(tf.browser.fromPixels($document[0].getElementById("dstCanvas"),3).expandDims()).arraySync();
    $log.log("Die Zahl ist: " + $scope.getMax(result[0]));
    $scope.zahl = $scope.getMax(result[0]);
    //Leere Canvas
    $scope.ctx.clearRect(0, 0, 224, 224);
    $scope.ctx.fillStyle = "white";
    $scope.ctx.fillRect(0, 0, 224, 224);
  }

  $scope.getMax = function(arr){
    let ret = [0,2,3,4,5,6,8];
    let max = -1000;
    let re = -1;
    arr.forEach((item, i) => {
      if (item > max){
        max = item;
        re = i;
      }
    });
    return ret[re];
  }

  // retrain the Model with the new Picture given
  $scope.retrain = function(idx){
    let t = new Array($scope.digits.length).fill(0);
    t[idx] = 1;
    // Set all the Buttons to disable!
    $scope.m = false;

    const features = tf.browser.fromPixels($document[0].getElementById("dstCanvas"),3).expandDims();
    const target = tf.tensor1d(t).expandDims();

    $scope.model.compile({loss: 'categoricalCrossentropy', metrics:['acc'], optimizer: 'adam'});
    $scope.model.fit(features, target, {epochs: 1, batchSize: 1,    callbacks: { onEpochEnd: async (epoch, logs) => {
      console.log("At Epoch: " + epoch + " Accuracy at:" + logs.acc);
    }}}).then(() =>{
      $scope.m = true;
      features.dispose();
      target.dispose();
    });
  }

  $scope.download_model = function(){
    $scope.model.save('downloads://my-model-new_' + new Date().getTime() / 1000);
  }

  /*FUNCTIONS END*/
  $scope.loadModel();
}]);
