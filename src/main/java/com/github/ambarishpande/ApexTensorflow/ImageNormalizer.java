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
  }


  public Tensor normalize(Data T)
  {
    try (Graph g = new Graph()) {
      ApexTensorflowScoringOperator.GraphBuilder b = new ApexTensorflowScoringOperator.GraphBuilder(g);

      final int H = 224;
      final int W = 224;
      final float mean = 117f;
      final float scale = 1f;

      // Since the graph is being constructed once per execution here, we can use a constant for the
      // input image. If the graph were to be re-used for multiple input images, a placeholder would
      // have been more appropriate.
//      final Output input = b.constant("input", imageBytes);
      final Output input = b.placeholder("input", T.bytesImage);

      final Output output =
        b.div(
          b.sub(
            b.resizeBilinear(
              b.expandDims(
                b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
                b.constant("make_batch", 0)),
              b.constant("size", new int[] {H, W})),
            b.constant("mean", mean)),
          b.constant("scale", scale));

      try (Session s = new Session(g)) {
        return s.runner().feed("input", Tensor.create( T.bytesImage)).fetch(output.op().name()).run().get(0);
      }
    }
  }
}
