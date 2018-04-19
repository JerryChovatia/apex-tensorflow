/**
 * Put your copyright and license info here.
 */

package com.github.ambarishpande.ApexTensorflow;
import org.apache.apex.api.plugin.DAGSetupPlugin;
import org.apache.hadoop.conf.Configuration;

import com.datatorrent.api.Context;
import com.datatorrent.api.annotation.ApplicationAnnotation;
import com.datatorrent.api.StreamingApplication;
import com.datatorrent.api.DAG;
import com.datatorrent.api.DAG.Locality;
import com.datatorrent.lib.io.ConsoleOutputOperator;

@ApplicationAnnotation(name="ApexTensorflowExample")
public class Application implements StreamingApplication
{

  @Override
  public void populateDAG(DAG dag, Configuration conf)
  {
    // Sample DAG with 2 operators
    // Replace this code with the DAG you want to build

    ImageReader imageReader = dag.addOperator("ImageReader",ImageReader.class);
    ImageNormalizer normalizer = dag.addOperator("Normalizer",ImageNormalizer.class);
    ApexTensorflowScoringOperator scorer = dag.addOperator("Scorer",ApexTensorflowScoringOperator.class);
    ConsoleOutputOperator console = dag.addOperator("Console", new ConsoleOutputOperator());

    dag.addStream("Image", imageReader.output, normalizer.input).setLocality(Locality.CONTAINER_LOCAL);
    dag.addStream("Normalized Image", normalizer.output, scorer.input).setLocality(Locality.CONTAINER_LOCAL);
    dag.addStream("Predicted Class",scorer.outputBestCategory,console.input);

    dag.setAttribute(scorer, Context.OperatorContext.METRICS_AGGREGATOR, new TfMetricsAggregator());
  }
}
