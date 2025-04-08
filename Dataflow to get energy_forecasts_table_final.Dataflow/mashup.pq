[DefaultOutputDestinationSettings = [DestinationDefinition = [Kind = "Reference", QueryName = "DefaultDestination", IsNewTarget = true], UpdateMethod = [Kind = "Replace"]], StagingDefinition = [Kind = "FastCopy"]]
section Section1;
shared DefaultDestination = Lakehouse.Contents(null){[workspaceId = "1cf87aea-aa0c-4a93-80a1-929e61be15e5"]}[Data]{[lakehouseId = "486921fd-3eb3-47c0-b9ce-f7804a355198"]}[Data];
[DataDestinations = {[Definition = [Kind = "Reference", QueryName = "energy_forecasts_DataDestination", IsNewTarget = true], Settings = [Kind = "Manual", AllowCreation = true, ColumnSettings = [Mappings = {[SourceColumnName = "Entity", DestinationColumnName = "Entity"], [SourceColumnName = "EnergyType", DestinationColumnName = "EnergyType"], [SourceColumnName = "Year", DestinationColumnName = "Year"], [SourceColumnName = "Forecast", DestinationColumnName = "Forecast"], [SourceColumnName = "Lower_CI", DestinationColumnName = "Lower_CI"], [SourceColumnName = "Upper_CI", DestinationColumnName = "Upper_CI"]}], DynamicSchema = false, UpdateMethod = [Kind = "Replace"], TypeSettings = [Kind = "Table"]]]}]
shared energy_forecasts = let
  Source = Lakehouse.Contents([]),
  #"Navigation 1" = Source{[workspaceId = "1cf87aea-aa0c-4a93-80a1-929e61be15e5"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "486921fd-3eb3-47c0-b9ce-f7804a355198"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "energy_forecasts", ItemKind = "Table"]}[Data],
  #"Changed column type" = Table.TransformColumnTypes(#"Navigation 3", {{"Year", type text}})
in
  #"Changed column type";
[BindToDefaultDestination = true]
shared modelling = let
  Source = Lakehouse.Contents([]),
  #"Navigation 1" = Source{[workspaceId = "1cf87aea-aa0c-4a93-80a1-929e61be15e5"]}[Data],
  #"Navigation 2" = #"Navigation 1"{[lakehouseId = "486921fd-3eb3-47c0-b9ce-f7804a355198"]}[Data],
  #"Navigation 3" = #"Navigation 2"{[Id = "modelling", ItemKind = "Table"]}[Data]
in
  #"Navigation 3";
shared energy_forecasts_DataDestination = let
  Pattern = Fabric.Warehouse([CreateNavigationProperties = false]),
  Navigation_1 = Pattern{[workspaceId = "1cf87aea-aa0c-4a93-80a1-929e61be15e5"]}[Data],
  Navigation_2 = Navigation_1{[warehouseId = "bd94930b-491b-4044-a79b-c21ca7f4b303"]}[Data],
  TableNavigation = Navigation_2{[Item = "energy_forecasts", Schema = "dbo"]}?[Data]?
in
  TableNavigation;
