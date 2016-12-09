var precomputedResults = null;
var commNames = ['OneToAll', 'TreeAllReduce', 'ButterflyAllReduce'];

// Gather inputs to paleo.
$(document).ready(function(){
  $('#paleo-control-info__strong').hide();

  $('#paleo-submit-btn').click(onSubmit);

  $('input[name=paleo-input__scaling]').change(function(){
    if ($(this).val() == 'weak') {
      $('#paleo-control-info__strong').hide();
      $('#paleo-control-info__weak').show();
    } else {
      $('#paleo-control-info__weak').hide();
      $('#paleo-control-info__strong').show();
    }
  });

  $('#paleo-input__batch_size').html(
    Math.pow(2, parseInt($('#paleo-input__batch_size_power').val())));
  $('#paleo-input__batch_size_power').change(function(){
    $('#paleo-input__batch_size').html(
      Math.pow(2, parseInt($('#paleo-input__batch_size_power').val())));
  });

  $.getJSON("precompute.json", function(json) {
    precomputedResults = json;
    onSubmit();
  }).fail(function() {
    console.error('Cannot load precomputed results.');
  });
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
  return meta.replace(/,/g, "<br>");
}

// Success. Plot the results.
function onPaleoSuccess(data) {
  var series = data.series;

  // Add absolute time to the tooltip description.
  var datasetSize = parseFloat($('#paleo-input__dataset_size').val());
  var nepoch = parseFloat($('#paleo-input__nepoch').val());
  var batch_size = parseFloat($('#paleo-input__batch_size').html());
  var iterations = datasetSize / batch_size * nepoch;
  for (var i = 0; i < data.series.length; i++) {
    for (var j = 0; j < series[i].length; j++) {
      var time = Math.round(iterations * data.times[i][j] / 1000 / 3600);
      var meta = [commNames[i],
        Math.round(series[i][j]) + "x speedup", time + ' hours'].join()
      series[i][j] = {meta: meta, value: series[i][j]};
    }
  }

  var data = {
    labels: data.labels,
    series: series,
  };

  new Chartist.Line('.ct-chart', data, plotOptions);
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
    onlyInteger: true
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
        axisTitle: 'Speedup relative to one worker',
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

new Chartist.Line('.ct-chart', data, plotOptions);
