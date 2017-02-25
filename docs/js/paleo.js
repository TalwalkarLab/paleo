var precomputedResults = null;
var commNames = ['OneToAll', 'TreeAllReduce', 'ButterflyAllReduce'];
var EC2_P2_COST_PER_GPU_PER_HOUR = 0.9;

// Gather inputs to paleo.
$(document).ready(function(){
  $('#paleo-control-info__strong').hide();

  $('#paleo-submit-btn').click(onSubmit);

  // Tips for weak / strong scaling.
  $('input[name=paleo-input__scaling]').change(function(){
    if ($(this).val() == 'weak') {
      $('#paleo-control-info__strong').hide();
      $('#paleo-control-info__weak').show();
    } else {
      $('#paleo-control-info__weak').hide();
      $('#paleo-control-info__strong').show();
    }
  });

  // Batch size slider.
  $('#paleo-input__batch_size').html(
    Math.pow(2, parseInt($('#paleo-input__batch_size_power').val())));
  $('#paleo-input__batch_size_power').on('change mousemove', function(){
    $('#paleo-input__batch_size').html(
      Math.pow(2, parseInt($('#paleo-input__batch_size_power').val())));
  });

  // Cloud selector.
  $('#paleo-input__cloud').change(function(){
    var cloud = $('#paleo-input__cloud').val();
    if (cloud == 'awsp2') {
      $('#paleo-input__device').val('K80');
      $('#paleo-input__network').val('ethernet20');
    }
  });
  var selectEC2 = function(){
    if ($('#paleo-input__device').val() == 'K80' &&
      $('#paleo-input__network').val() == 'ethernet20'){
      $('#paleo-input__cloud').val('awsp2');
    } else {
      $('#paleo-input__cloud').val('none');
    }
  }
  $('#paleo-input__device').change(selectEC2);
  $('#paleo-input__network').change(selectEC2);

  // Load results.
  $.getJSON("precompute.json", function(json) {
    precomputedResults = json;
    onSubmit();
  }).fail(function() {
    console.error('Cannot load precomputed results.');
  });

  // Auto submit.
  $('.paleo-input-knob').on('change', onSubmit);
});

function inputsToKey(paleoInputs){
  var e = [
    paleoInputs.model, paleoInputs.device, paleoInputs.network,
    paleoInputs.software, paleoInputs.batch_size, paleoInputs.scaling,
    paleoInputs.use_cudnn];
  return e.join();
}

// Collect inputs to Paleo from the form.
function gatherPaleoInputs() {
  var paleoInputs = {};
  paleoInputs.model = $('#paleo-input__model').val();
  paleoInputs.device = $('#paleo-input__device').val();
  paleoInputs.network = $('#paleo-input__network').val();
  paleoInputs.software = $('#paleo-input__software').val();
  paleoInputs.batch_size = parseInt($('#paleo-input__batch_size').html());
  paleoInputs.scaling = $('input[name=paleo-input__scaling]:checked').val();
  paleoInputs.use_cudnn = $('#paleo-input__use_cudnn').is(':checked');
  return paleoInputs;
}

function tooltipFunc(meta, value){
  return meta.replace(/;/g, "<br>");
}

// Success. Plot the results.
function onPaleoSuccess(data) {

  var seriesSpeedup = [];
  for (var i = 0; i < data.series.length; i++) {
    seriesSpeedup.push(data.series[i].map(function(speedup){
      var meta = [commNames[i], Math.round(speedup) + "x speedup"].join(';')
      return {meta: meta, value: speedup};
    }));
  }

  new Chartist.Line('.ct-chart-speedup',
    {labels: data.labels, series: seriesSpeedup}, plotOptions);


  // Time plot
  var seriesTime = [];
  var datasetSize = parseFloat($('#paleo-input__dataset_size').val());
  var nepoch = parseFloat($('#paleo-input__nepoch').val());
  var batch_size = parseFloat($('#paleo-input__batch_size').html());
  var iterations = datasetSize / batch_size * nepoch;
  for (var i = 0; i < data.times.length; i++) {
    seriesTime.push(data.times[i].map(function(t){
      var timeInHour = Math.ceil(iterations * t / 1000 / 3600);
      var meta = [commNames[i], timeInHour.toLocaleString() + " hours"].join(';');
      return {meta: meta, value: timeInHour};
    }));
  }
  new Chartist.Line('.ct-chart-time',
    {labels: data.labels, series: seriesTime}, timePlotOptions);

  // Cost plot
  var seriesCost = [];
  var cloud = $('#paleo-input__cloud').val();
  if (cloud == 'awsp2') {
    for (var i = 0; i < data.times.length; i++) {
      seriesCost.push(data.times[i].map(function(t, index){
        var timeInHour = Math.ceil(iterations * t / 1000 / 3600);
        var ngpus = Math.pow(2, index);
        var cost = Math.round(
          timeInHour * EC2_P2_COST_PER_GPU_PER_HOUR * ngpus);
        var meta = [commNames[i], "$" + cost.toLocaleString()].join(';');
        return {meta: meta, value: cost};
      }));
    }
  }
  new Chartist.Line('.ct-chart-cost',
      {labels: data.labels, series: seriesCost}, costPlotOptions);
}

function onSubmit(e) {
  var paleoInputs = gatherPaleoInputs();
  var key = inputsToKey(paleoInputs);
  if (precomputedResults === null){
    var notification = document.querySelector('.mdl-js-snackbar');
    notification.MaterialSnackbar.showSnackbar({
        message: "Precomputed results are not loaded."
    });
    return;
  }
  if (precomputedResults[key] === undefined) {
    var notification = document.querySelector('.mdl-js-snackbar');
    notification.MaterialSnackbar.showSnackbar({
        message: "Oops, Paleo did not precompute this configuration."
    });
    return;
  }

  onPaleoSuccess(precomputedResults[key]);
}


// Chartist
var data = {
  labels: ['1', '2', '4', '8', '16', '32', '64', '128'],
  series: []
};

var plotOptions = {
  height: 300,
  fullWidth: true,
  chartPadding: {
    top: 10,
    right: 30,
    bottom: 30,
    left: 20
  },
  axisY: {
    onlyInteger: true,
    // type: Chartist.AutoScaleAxis,
    // scale: 'log10',
  },
  plugins: [
    Chartist.plugins.tooltip({
      anchorToPoint: false,
      appendToBody: true,
      tooltipFnc: tooltipFunc
    }),
    Chartist.plugins.legend({
      legendNames: ['OneToAll', 'TreeAllReduce', 'ButterflyAllReduce'],
    }),
    Chartist.plugins.ctAxisTitle({
      axisX: {
        axisTitle: 'Number of Workers',
        axisClass: 'ct-axis-title',
        offset: {
          x: 0,
          y: 40
        },
        textAnchor: 'middle'
      },
      axisY: {
        axisTitle: 'Throughput speedup',
        axisClass: 'ct-axis-title',
        offset: {
          x: 0,
          y: 20
        },
        textAnchor: 'middle',
        flipTitle: true
      }
    })
  ]
};

var timePlotOptions = $.extend(true, {}, plotOptions);
timePlotOptions.plugins[2] = Chartist.plugins.ctAxisTitle({
  axisX: {
    axisTitle: 'Number of Workers',
    axisClass: 'ct-axis-title',
    offset: {
      x: 0,
      y: 40
    },
    textAnchor: 'middle'
  },
  axisY: {
    axisTitle: 'Training time (hours)',
    axisClass: 'ct-axis-title',
    offset: {
      x: 0,
      y: 10
    },
    textAnchor: 'middle',
    flipTitle: true
  }
});

var costPlotOptions = $.extend(true, {}, plotOptions);
costPlotOptions.plugins[2] = Chartist.plugins.ctAxisTitle({
  axisX: {
    axisTitle: 'Number of Workers',
    axisClass: 'ct-axis-title',
    offset: {
      x: 0,
      y: 40
    },
    textAnchor: 'middle'
  },
  axisY: {
    axisTitle: 'Cost (USD)',
    axisClass: 'ct-axis-title',
    offset: {
      x: 0,
      y: 10
    },
    textAnchor: 'middle',
    flipTitle: true
  }
});

new Chartist.Line('.ct-chart-speedup', data, plotOptions);
new Chartist.Line('.ct-chart-time', data, timePlotOptions);
new Chartist.Line('.ct-chart-cost', data, costPlotOptions);
