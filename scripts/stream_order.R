library("nhdplusTools")
source(system.file("extdata", "walker_data.R", package = "nhdplusTools"))
# IN_flowline <- 
test_flowline <- prepare_nhdplus(walker_flowline, 0, 0, FALSE)
test_flowline <- data.frame(
  ID = test_flowline$COMID,
  toID = test_flowline$toCOMID)
(order <- get_streamorder(test_flowline))
walker_flowline$order <- order
plot(sf::st_geometry(walker_flowline), lwd = walker_flowline$order, col = "blue")

# ###
# gdb <- "C:\\Users\\jmatney\\Documents\\GitHub\\IndianaRisk\\data\\NHDPLUS_H_0512_HU4_GDB.gdb"
# IN_flowline <- sf::read_sf(dsn=gdb, layer="NHDFlowline")
# IN_flowline_test <- prepare_nhdplus(IN_flowline, 0, 0, FALSE)
# str(IN_flowline)
