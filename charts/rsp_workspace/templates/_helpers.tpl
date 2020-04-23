{{/*https://github.com/openshift/origin/issues/24060*/}}
{{- define "chart.helmRouteFix" -}}
status:
  ingress:
    - host: ""
{{- end -}}
