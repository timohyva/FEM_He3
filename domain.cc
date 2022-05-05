/* ---------------------------------------------------------------------
 *
 * dell.II libs based FEM codes for seearching order parameters field 
 * distribution in superfluid helium-3
 * 
 * author: Quang (timohyva@github)
 *
 * ---------------------------------------------------------------------

 */



// The most fundamental class in the library is the Triangulation class, which
// is declared here:
#include <deal.II/grid/tria.h>

#include <deal.II/base/point.h>

// Here are some functions to generate standard grids:
#include <deal.II/grid/grid_generator.h>
// Output of grids in various graphics formats:
#include <deal.II/grid/grid_out.h>

// This is needed for C++ output:
#include <iostream>
#include <fstream>
// And this for the declarations of the `std::sqrt` and `std::fabs` functions:
#include <cmath>

// we simply import the entire deal.II
// namespace for general use:
using namespace dealii;

// ************************************************************************



// ************************************************************************

void merged_grid()
{
  Triangulation<2> recta1, recta2, recta3;
 

  // fill it with retangular shape 1
  const Point<2> upper_left1(-4., 2.);
  const Point<2> lower_right1(-2.5, -2.);
  GridGenerator::hyper_rectangle(recta1, upper_left1, lower_right1);
  
  // fill it with retangular shape 2
  const Point<2> upper_left2(-2.5, 0.2);
  const Point<2> lower_right2(2.5, -0.2);
  GridGenerator::hyper_rectangle(recta2, upper_left2, lower_right2);

  // fill it with retangular shape 2
  const Point<2> upper_left3(2.5, 2.);
  const Point<2> lower_right3(4., -2.);
  GridGenerator::hyper_rectangle(recta3, upper_left3, lower_right3);

  
  Triangulation<2> merged_domain;
  GridGenerator::merge_triangulations({&recta1, &recta2, &recta3},
                                    merged_domain,
                                    1.0e-10,
                                    false);

   merged_domain.refine_global(1);
  // five times refinements for boundaries
  for (unsigned int step = 0; step < 2; ++step)
    {
      for (auto &cell : merged_domain.active_cell_iterators())
	{
         for (const auto v : cell->vertex_indices())
	   {
	     std::cout << " v is " << v << " vertex looks like " << cell->vertex(v) << std::endl;

	     double y_vertex = (cell->vertex(v)).operator()(1);
	     double x_vertex = (cell->vertex(v)).operator()(0);
	     if (
		  ((std::fabs(y_vertex - upper_left2.operator()(1))  <= 1e-6 * std::fabs(upper_left2.operator()(1)))
		   || (std::fabs(y_vertex - lower_right2.operator()(1))  <= 1e-6 * std::fabs(lower_right2.operator()(1))))
		  && ((x_vertex >= upper_left2.operator()(0)) && (x_vertex <= lower_right2.operator()(0)))
		  
		  || (std::fabs(x_vertex) == lower_right2.operator()(0))
		   
		 )
	       {
		cell->set_refine_flag();
                break;		 
	       }

	   }
	  
	}

      merged_domain.execute_coarsening_and_refinement();
    }
   merged_domain.refine_global(2);
  
  std::ofstream out("domain_merge.eps");
  GridOut       grid_out;
  grid_out.write_svg(merged_domain, out);

  std::cout << "Grid written to domain_merge.eps" << std::endl;
  
}

void channel_grid()
{
  // triangulation object for channel
  Triangulation<2> channel;

  // fill it with retangular shape
  const Point<2> upper_left(-2., 1.);
  const Point<2> lower_right(2., -1.);
  GridGenerator::hyper_rectangle(channel, upper_left, lower_right);

  
  // five times refinements for boundaries
  for (unsigned int step = 0; step < 6; ++step)
    {
      for (auto &cell : channel.active_cell_iterators())
	{
         for (const auto v : cell->vertex_indices())
	   {
	     std::cout << " v is " << v << " vertex looks like " << cell->vertex(v) << std::endl;

	     double y_component_vertex = (cell->vertex(v)).operator()(1);
	     if ((std::fabs(y_component_vertex - upper_left.operator()(1))  <= 1e-6 * std::fabs(upper_left.operator()(1)))
		 || (std::fabs(y_component_vertex - lower_right.operator()(1))  <= 1e-6 * std::fabs(lower_right.operator()(1))))
	       {
                cell->set_refine_flag();
                break;		 
	       }

	   }
	  
	}

      channel.execute_coarsening_and_refinement();
    }
  
  // global refinment for main body
  channel.refine_global(2);

  std::ofstream domain_plot("channel.eps");
  GridOut grid_out;
  grid_out.write_svg(channel, domain_plot);

  std::cout << " channel plot is randered to channel.eps " << std::endl;


}

// ****************************************************************************
// ***      main()

// Finally, the main function. There isn't much to do here, only to call the
// two subfunctions, which produce the two grids.
int main()
{
 
  channel_grid();

  merged_grid();

  return 0;
}
