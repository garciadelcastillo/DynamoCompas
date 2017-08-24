using System.Collections.Generic;
using Autodesk.DesignScript.Interfaces;
using Dynamo.Graph.Nodes;
using Autodesk.DesignScript.Runtime;

/*
 Autodesk ETH Workshop Zurich
 August 2017
 */

namespace Compas.Dynamo.Datastructures
{
    // Add the IsDesignScriptCompatible attribute to ensure
    // that it gets loaded in Dynamo.
    [IsDesignScriptCompatible]
    public class CompasNetwork : IGraphicItem
    {
        #region private members

        // IronPython mesh object 
        private object pythonNetwork;
        // string representation of this mesh (brought from python)
        private string str;
        // [[x,y,z],[x,y,z],[..]]
        private List<object> vertices;
        // [[x,y,z],[x,y,z],[..]]
        private List<object> edges;

        #endregion

        #region properties

        // Nothing here yet =).

        #endregion

        #region public methods

        private CompasNetwork(object _pythonNetwork, string stringRepresentation, List<object> _vertices, List<object> _indices)
        {
            str = stringRepresentation;
            pythonNetwork = _pythonNetwork;
            vertices = _vertices;
            edges = _indices;
        }

        [CanUpdatePeriodically(true)]
        public static CompasNetwork Create(object pythonMesh, string stringRepresentation, List<object> vertices, List<object> indices)
        {
            return new CompasNetwork(pythonMesh, stringRepresentation, vertices, indices);
        }

        #endregion

        #region IGraphicItem interface


        /// <summary>
        /// The Tessellate method in the IGraphicItem interface allows
        /// you to specify what is drawn when dynamo's visualization is
        /// updated.
        /// </summary>
        [IsVisibleInDynamoLibrary(false)]
        public void Tessellate(IRenderPackage package, TessellationParameters parameters)
        {
            // Vertices
            if (vertices != null)
            {
                foreach (List<object> p in vertices)
                {
                    double x = (double)p[0];
                    double y = (double)p[1];
                    double z = (double)p[2];
                    package.AddPointVertex(x, y, z);
                    package.AddPointVertexColor(255, 0, 0, 255);
                }
            }

            // Faces (for CompasMesh)
            //foreach (List<object> ig in indices)
            //{
            //    if (ig.Count == 3)
            //    {
            //        DrawColoredLine(package, pts[ig.A], pts[ig.B], zHeights[ig.A], zHeights[ig.B]);
            //        DrawColoredLine(package, pts[ig.B], pts[ig.C], zHeights[ig.B], zHeights[ig.C]);
            //        DrawColoredLine(package, pts[ig.C], pts[ig.A], zHeights[ig.C], zHeights[ig.A]);
            //    }
            //    else if (ig.Count == 4)
            //    {
            //        DrawColoredLine(package, pts[ig.A], pts[ig.B], zHeights[ig.A], zHeights[ig.B]);
            //        DrawColoredLine(package, pts[ig.B], pts[ig.C], zHeights[ig.B], zHeights[ig.C]);
            //        DrawColoredLine(package, pts[ig.C], pts[ig.D], zHeights[ig.C], zHeights[ig.D]);
            //        DrawColoredLine(package, pts[ig.D], pts[ig.A], zHeights[ig.D], zHeights[ig.A]);
            //    }
            //}
        }

        #endregion

        public object ToPythonNetwork()
        {
            return this.pythonNetwork;
        }

        public override string ToString()
        {
            //return string.Format("{0}", this.str);
            return this.str;
        }
    }
}
