using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Autodesk.DesignScript.Runtime;
using Autodesk.DesignScript.Interfaces;
using Autodesk.DesignScript.Geometry;

namespace DynamoCompas
{

    [IsVisibleInDynamoLibrary(false)]
    public class CompasNetworkWrapper : IGraphicItem
    {
        private Mesh mesh;
        private double minZ = Double.MaxValue,
                       maxZ = Double.MinValue,
                       dz;


        public CompasNetworkWrapper(Mesh mesh)
        {
            this.mesh = mesh;

            int i = 0;
            foreach (Point p in mesh.VertexPositions)
            {
                if (p.Z < minZ) minZ = p.Z;
                if (p.Z > maxZ) maxZ = p.Z;
            }
            dz = maxZ - minZ;
        }


        public void Tessellate(IRenderPackage package, TessellationParameters parameters)
        {
            Point[] pts = mesh.VertexPositions;
            Vector[] normals = mesh.VertexNormals;
            IndexGroup[] indices = mesh.FaceIndices;

            bool foo = package.DisplayLabels;

            int i = 0;
            foreach (Point p in pts)
            {
                package.AddPointVertex(p.X, p.Y, p.Z);
                package.AddPointVertexColor(255, 0, 0, 255);
            }

            foreach (IndexGroup ig in indices)
            {
                if (ig.Count == 3)
                {
                    DrawColoredLine(package, this, pts[ig.A], pts[ig.B]);
                    DrawColoredLine(package, this, pts[ig.B], pts[ig.C]);
                    DrawColoredLine(package, this, pts[ig.C], pts[ig.A]);
                }
                else if (ig.Count == 4)
                {
                    DrawColoredLine(package, this, pts[ig.A], pts[ig.B]);
                    DrawColoredLine(package, this, pts[ig.B], pts[ig.C]);
                    DrawColoredLine(package, this, pts[ig.C], pts[ig.D]);
                    DrawColoredLine(package, this, pts[ig.D], pts[ig.A]);
                }
            }
        }

        internal static void DrawColoredLine(IRenderPackage package, CompasNetworkWrapper network, Point p0, Point p1)
        {
            package.AddLineStripVertex(p0.X, p0.Y, p0.Z);
            package.AddLineStripVertex(p1.X, p1.Y, p1.Z);

            byte blue0 = (byte)Math.Round(255 * ((p0.Z - network.minZ) / network.dz));
            byte blue1 = (byte)Math.Round(255 * ((p1.Z - network.minZ) / network.dz));

            package.AddLineStripVertexColor(0, 0, blue0, 255);
            package.AddLineStripVertexColor(0, 0, blue1, 255);

            // Specify line segments by adding a line vertex count.
            // Ex. The above line has two vertices, so we add a line
            // vertex count of 2. If we had tessellated a curve with n
            // vertices, we would add a line vertex count of n.
            package.AddLineStripVertexCount(2);
        }

    }
}
