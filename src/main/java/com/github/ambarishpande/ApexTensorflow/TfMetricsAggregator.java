package com.github.ambarishpande.ApexTensorflow;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;

import com.google.common.collect.Maps;

import com.datatorrent.api.AutoMetric;
import com.datatorrent.common.util.Pair;

/**
 * Created by ambarish on 21/9/17.
 */
public class TfMetricsAggregator implements AutoMetric.Aggregator, Serializable
{
  Map<String, Object> result = Maps.newHashMap();

  @Override
  public Map<String, Object> aggregate(long l, Collection<AutoMetric.PhysicalMetricsContext> collection)
  {

    Collection<Collection<Pair<String, Object>>> ret = new ArrayList<>();
    for (AutoMetric.PhysicalMetricsContext pmc : collection) {
      for (Map.Entry<String, Object> metrics : pmc.getMetrics().entrySet()) {
        String key = metrics.getKey();
        Object value = metrics.getValue();
        switch (key) {
          case "labelCounts":
            Collection<Collection<Pair<String, Object>>> temp = (Collection<Collection<Pair<String, Object>>>)value;
            ret.addAll(temp);
            break;
        }
      }
    }

    if (ret.size() > 0) {
      result.put("Label-Counts", ret);
    }


    return result;
  }
}

