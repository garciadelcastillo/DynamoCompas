using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Autodesk.DesignScript.Interfaces;
using Dynamo.Graph.Nodes;
using Autodesk.DesignScript.Runtime;


namespace Compas.Dynamo.Datastructures
{
    [IsDesignScriptCompatible]
    public class CompassNetworkAnimationViz : IGraphicItem
    {
        private CompasNetwork network;
        private int it;
        private int frameCount;

        internal CompassNetworkAnimationViz(CompasNetwork network, int it)
        {
            this.network = network;
            this.it = it;
            this.frameCount = network.frameDoubles.Length;
        }


        public static CompassNetworkAnimationViz AnimateNetwork(CompasNetwork network, int iteration = 0)
        {
            return network == null ? null : new CompassNetworkAnimationViz(network, iteration);
        }


        [IsVisibleInDynamoLibrary(false)]
        public void Tessellate(IRenderPackage package, TessellationParameters parameters)
        {
            if (network.frames != null)
            {
                double[][] frame = network.frameDoubles[it % frameCount];
                CompasNetwork.RenderNetworkFrame(network, frame, package, parameters);
            }
        }

        
    }
}
