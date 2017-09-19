package com.github.ambarishpande.ApexTensorflow;

import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import com.datatorrent.api.Context;
import com.datatorrent.api.DefaultInputPort;
import com.datatorrent.api.DefaultOutputPort;
import com.datatorrent.common.util.BaseOperator;

/**
 * Created by ambarish on 18/9/17.
 */
public class ImageNormalizer extends BaseOperator
{

  private Graph g;
  private ApexTensorflowScoringOperator.GraphBuilder b;

  private transient Output inputPlaceholder;
  private transient Output outputComputation;

  public  transient DefaultOutputPort<Image> output = new DefaultOutputPort<>();
  public  transient DefaultInputPort<Data> input = new DefaultInputPort<Data>()
  {
    @Override
    public void process(Data data)
    {
      Image i = new Image(normalize(data),data);
      output.emit(i);
    }
  };

  @Override
  public void setup(Context.OperatorContext context)
  {
    super.setup(context);
    initializeComputations();
  }

  private void initializeComputations(){
    g = new Graph();
    ApexTensorflowScoringOperator.GraphBuilder b = new ApexTensorflowScoringOperator.GraphBuilder(g);

    final int H = 224;
    final int W = 224;
    final float mean = 117f;
    final float scale = 1f;


    inputPlaceholder = b.placeholder("input");

    outputComputation =
      b.div(
        b.sub(
          b.resizeBilinear(
            b.expandDims(
              b.cast(b.decodeJpeg(inputPlaceholder, 3), DataType.FLOAT),
              b.constant("make_batch", 0)),
            b.constant("size", new int[] {H, W})),
          b.constant("mean", mean)),
        b.constant("scale", scale));
  }


  public Tensor normalize(Data T)
  {
      try (Session s = new Session(g)) {
        return s.runner().feed("input", Tensor.create( T.bytesImage)).fetch(outputComputation.op().name()).run().get(0);
      }
  }
}
